# NBME - Score Clinical Patient Notes

![项目示意图](PNG/NBME_photos.png)

[Kaggle Competition: NBME - Score Clinical Patient Notes](https://www.kaggle.com/c/nbme-score-clinical-patient-notes)

## 项目简介

这是一个围绕 Kaggle `NBME - Score Clinical Patient Notes` 竞赛构建的医疗文本信息抽取项目。任务目标是根据 `feature_text`，从临床病历 `patient notes` 中定位对应的证据片段，并输出字符级别的 `span` 位置。

这是个人的银牌解决方案

从建模角度看，这不是传统的分类任务，而是一个更接近 **token classification / span extraction** 的命名实体识别问题。项目核心做法是：

- 使用 `DeBERTa` 系列预训练模型作为 backbone
- 先在 `patient_notes.csv` 上进行领域自适应 `Masked Language Modeling (MLM)` 预训练
- 再进行 `5-fold` 监督微调，学习 `feature_text -> note span` 的对齐关系
- 使用 pseudo labeling 扩充训练数据
- 在推理阶段进行多模型、多折平均与加权融合

这套流程覆盖了从预训练、监督训练、伪标签生成、全量再训练到最终提交推理的完整竞赛闭环。

## 任务定义

给定一条样本：

- `feature_text`：需要识别的医学要点或症状描述
- `pn_history`：患者病历文本

模型需要从 `pn_history` 中抽取与 `feature_text` 对应的文本片段，并输出为字符区间，例如：

```text
["203 217", "320 336"]
```

竞赛评估指标为 **span-level micro F1**。因此，这个项目特别强调：

- token 到 character 的精确对齐
- offset mapping 的稳定处理
- 从 token logits 还原到字符级概率
- 最终 span 后处理的一致性

## 数据集说明

仓库当前包含比赛使用的核心表：

- `datasets/train.csv`：训练标注，包含 `annotation` 与 `location`
- `datasets/features.csv`：待抽取医学概念 `feature_text`
- `datasets/patient_notes.csv`：原始临床病历文本
- `datasets/test.csv`：测试集
- `datasets/sample_submission.csv`：提交格式示例
- `datasets/train_processed.pkl`：处理后的训练数据
- `datasets/train_pl_all.pkl`：用于伪标签预测的扩展数据

原始表结构可概括为：

- `train.csv`: `id`, `case_num`, `pn_num`, `feature_num`, `annotation`, `location`
- `features.csv`: `feature_num`, `case_num`, `feature_text`
- `patient_notes.csv`: `pn_num`, `case_num`, `pn_history`

## 方法概览

整体方案可以分为 5 个阶段：

1. 领域自适应预训练
2. 监督式 5 折训练
3. 伪标签生成
4. 加入伪标签后的再训练 / finetune
5. 多模型融合推理

### 1. 领域自适应预训练

文件：`1.nbme-pretrain.ipynb`

做法是在 `patient_notes.csv` 的 `pn_history` 上，对 `microsoft/deberta-base` 做 MLM 继续预训练，以便模型先适应临床病历中的缩写、拼写噪声和医学表达习惯。

关键配置：

- backbone: `microsoft/deberta-base`
- epoch: `10`
- batch size: `32`
- learning rate: `1e-5`
- mlm probability: `0.2`
- 输出实验 ID: `1001`

这个阶段产出的 checkpoint 被后续监督训练直接复用。

### 2. 监督式 span/token 训练

文件：`2.nbme-train.ipynb`

该阶段将任务建模为 token-level 二分类。输入形式是：

```text
[patient note] + [feature text]
```

模型对每个 token 预测其是否属于目标证据 span。

实现细节：

- 使用 `AutoTokenizer(..., trim_offsets=False)`，尽量保留稳定的字符偏移
- 根据 `location` 标注把 char-level span 映射到 token-level label
- 损失函数为 `BCEWithLogitsLoss`
- 使用 `5-fold` 交叉验证训练
- 验证时将 token logits 映射回 char logits，再恢复 span 并计算 micro F1

关键配置：

- 预训练权重：`/output/1001/checkpoint-7908`
- 实验 ID：`1002`
- epoch: `10`
- batch size: `32`
- learning rate: `1e-5`
- folds: `5`

### 3. Pseudo Label 生成

文件：

- `3.nbme-pseudo-prediction.ipynb`
- `4.nbme-pseudo-blend.ipynb`

流程分两步：

- 用 `1002` 训练出的 5 折模型，对 `train_pl_all.pkl` 做预测，保存每条样本的 char-level logits
- 将多个模型输出进行加权融合，生成伪标签的 `location` 与 `annotation`

从 notebook 中可以看到，伪标签融合权重示例为：

```python
blend_weights = {
    "/home/xm/workspace/output/1002": 0.19,
    "/home/xm/workspace/output/1012": 0.37,
    "/home/xm/workspace/output/1022": 0.44,
}
```

融合后生成新的伪标签文件：

- `train_pl_1002.pkl`

### 4. 加入伪标签后的再训练

文件：

- `5.nbme-alldata-train.ipynb`
- `6.nbme-finetune.ipynb`

这两个 notebook 分别承担：

- 使用真实训练集 + 对应 fold 的 pseudo data 做再训练
- 在已有模型基础上继续短轮次 finetune

其中 `5.nbme-alldata-train.ipynb` 的训练逻辑是：

- 验证集仍使用当前 fold 的真实数据
- 训练集使用其余 fold 的真实数据
- 再拼接当前 fold 对应的 pseudo label 数据

关键配置：

- `1003`: 基于 `1002` 权重继续训练，epoch `3`
- `1004`: 基于 `1002` 权重再做 finetune，epoch `2`

这类设置通常有助于在不破坏交叉验证结构的前提下，引入更多弱监督样本信息。

### 5. 多模型融合推理

文件：`7.nbme-inference.ipynb`

最终推理阶段采用了：

- 多 backbone 融合
- 每个 backbone 下做 `5-fold` 预测平均
- 再对不同模型进行加权融合

notebook 中的推理模型配置为：

```python
CHECKPOINTS = [
    "../input/deberta-v3-large",
    "../input/microsoft-deberta-large",
    "../input/debertabase",
]

PATHS = [
    "../input/my-deberta-v3-large",
    "../input/my-deberta-large",
    "../input/my-deberta-base",
]

WEIGHTS = {
    "w0": 0.6,
    "w1": 0.25,
    "w2": 0.15,
}
```

推理输出流程：

- 每个模型对测试集生成 token logits
- 对 5 个 fold 的 logits 取平均
- 再映射成 char-level logits
- 多模型加权求和
- 根据阈值恢复 span
- 生成 `submission.csv`

## 模型结构

项目中的主模型非常简洁，属于典型的 `Transformer encoder + token head` 结构：

- backbone: `AutoModel`
- dropout: `0.1`
- classifier: `Linear(hidden_size, 1)`
- loss: `BCEWithLogitsLoss`

也就是说，竞赛提升主要不依赖复杂 head，而更多来自：

- 更合适的预训练策略
- 精准的 char/token 对齐
- 稳定的交叉验证
- 伪标签增强
- 多模型融合

## 文件说明

仓库中的 notebook 可以按执行顺序理解为：

- `1.nbme-pretrain.ipynb`：在病历文本上做 MLM 预训练
- `2.nbme-train.ipynb`：5-fold 监督训练，得到基础模型
- `3.nbme-pseudo-prediction.ipynb`：对扩展数据集生成伪标签 logits
- `4.nbme-pseudo-blend.ipynb`：融合伪标签结果并构造 pseudo dataset
- `5.nbme-alldata-train.ipynb`：真实数据 + 伪标签数据联合训练
- `6.nbme-finetune.ipynb`：进一步短轮次 finetune
- `7.nbme-inference.ipynb`：多模型融合推理并导出提交文件

## 复现建议

这个仓库不是单 notebook 演示，而是一套有先后依赖关系的竞赛 pipeline，建议严格按下面顺序复现：

1. 先准备 `datasets/train.csv`、`datasets/features.csv`、`datasets/patient_notes.csv`、`datasets/test.csv`、`datasets/sample_submission.csv`，并直接使用仓库里的 `train_processed.pkl` 作为清洗后的有标签训练集。
2. 运行 `1.nbme-pretrain.ipynb`，在全部 `patient notes` 上做 MLM 预训练。这个阶段对应 `exp_id=1001`，输出后续监督训练要用的 checkpoint。
3. 运行 `2.nbme-train.ipynb`，基于预训练权重做 5-fold 监督训练。这里会保存每个 fold 的 `*.pt` 权重、`config.json` 和 `oof.pkl`，是后续 pseudo label 的输入。
4. 如果只想先复现单模型 baseline，到这里就可以停。若要复现完整高分方案，继续运行 `3.nbme-pseudo-prediction.ipynb`，对 `train_pl_all.pkl` 生成 char-level logits。
5. 运行 `4.nbme-pseudo-blend.ipynb`，把多个模型的伪标签结果按权重融合，生成新的 pseudo 数据文件。PDF 里给出的伪标签融合权重是 `base: 0.19`、`large: 0.37`、`v3-large: 0.44`。
6. 运行 `5.nbme-alldata-train.ipynb`，把真实标签数据和 pseudo label 数据合并继续训练。代码里采用的是“当前 fold 验证集保持真实标签，训练集拼接对应 fold 的 pseudo data”这一套防泄漏方式。
7. 运行 `6.nbme-finetune.ipynb`，再回到正式标注样本上做少量 epoch 的 finetune，得到最终可用于提交的 fold 权重。
8. 最后运行 `7.nbme-inference.ipynb`。这个 notebook 会对每个 backbone 做 5-fold 平均，再做模型间加权融合，输出 `submission.csv`。PDF 里的最终推理权重是 `base: 0.15`、`large: 0.25`、`v3-large: 0.6`。

复现时最需要先改的是路径。当前 notebook 同时保留了本地路径 `/home/xm/workspace/...` 和 Kaggle 路径 `/input/...`，不先统一路径，后面的训练和推理不会顺着跑通。

## 项目亮点

- 这不是普通的单模型 finetune 仓库，而是完整实现了 `Pretrain -> Train -> Pseudo Label -> All Data Train -> Finetune -> Inference` 的竞赛闭环，和 `readme.pdf` 中描述的主方案一致。
- 数据侧没有直接吃官方原始标注，而是显式使用了修复后的 `datasets/train_processed.pkl`。这一点很关键，因为 PDF 里也提到官方标签里有脏标注，项目实际代码已经把这部分处理纳入训练入口。
- 监督训练和推理都围绕 span 级指标设计：训练时把 `location` 从 char-level 映射到 token label，验证和提交时再把 token logits 回写到 char logits，最后恢复成 `location span`。这比只报 token F1 更贴比赛本身。
- Pseudo label 不是简单把无标签样本全塞进训练，而是先对 `train_pl_all.pkl` 逐 fold 预测，再做模型融合，再按 fold 拼回训练集，代码里专门避免了验证泄漏。
- 项目实际用了 3 个 DeBERTa backbone：`deberta-base`、`deberta-large`、`deberta-v3-large`。最终不是只做单模平均，而是先做 fold 内平均，再做模型间加权融合，这也是 PDF 里明确写出的最终提交策略。
- 按 PDF 记录，这套方案最终拿到了 `Private LB 0.883`，处于 `Top 1%`；Public LB 经过 pretrain、三模型融合和 pseudo label 调权后提升到 `0.893+`。这说明仓库里的各个 notebook 不是练手脚本，而是对应过真实有效的竞赛迭代。

