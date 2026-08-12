# MFON多模态情感分析项目上下文

> 最后更新：2026-08-10
>
> 本文件是新Codex对话的首要入口。不要从零开始，不要把未完成实验写成已证明结论。
> 当前MOSEI服务器任务请优先读取根目录 `MOSEI-HANDOFF-LATEST.md`。

## 1. 项目位置

- 本地工作区：`/Users/augustus/projects/论文`
- 本地代码：`/Users/augustus/projects/论文/MFON`
- 服务器代码：`/home/jovyan/projects/MFON`
- 已冻结数据集：CMU-MOSI；当前跨数据集阶段：CMU-MOSEI
- 基础模型：MFON（COLING 2025）
- 当前方法暂称：审计驱动的干预式可靠性预算学习
- 旧名`Q-DAMFON`已停用；不把MFON基础结构宣称为原创。

## 2. 当前方法

1. 视觉和音频分别学习逐样本可靠性。
2. 为每个样本生成clean/mild/strong人工退化三联对，使可靠性分数随退化强度下降。
3. 音频使用专用时序头，描述相邻帧差分、差分能量、滞后相关和峰度。
4. 将MFON的KL蒸馏和InfoNCE保留为逐样本损失。
5. 每个mini-batch的辅助监督平均预算严格固定，仅根据可靠性重新分配给不同样本。
6. P4最终候选保持主情感任务输入干净；可靠性头单独从人工退化中学习。
7. 辅助监督平均预算从第1轮起固定为0.5，只将样本间分配从均匀逐步过渡到可靠性感知。
8. 作用性对照保持模型、退化、可靠性监督和平均预算一致，只改变分数与样本的对应方式。

## 3. 已完成修复

- 去掉冻结编码器输出的`.squeeze()`，防止batch size 1丢维。
- `position_embedding.py`改用`torch.arange`生成离散位置，修复`index_select`越界。
- 固定预算、可靠性、退化审计与P2对照共25项CPU测试通过。

## 4. 主要基线

MOSI seed 1111修复版MFON：

```text
Has0 Acc-2=0.8265, Has0 F1=0.8253
Non0 Acc-2=0.8476, Non0 F1=0.8471
Acc-5=0.5029, Acc-7=0.4388
MAE=0.7279, Corr=0.7959
```

修复版MFON三种子：

```text
Has0 Acc-2=0.8270±0.0009
Acc-7=0.4505±0.0125
MAE=0.7258±0.0028
Corr=0.7943±0.0015
```

## 5. 可靠性审计

25轮 learned 最佳checkpoint，MOSI完整测试集686样本：

```text
vision: Spearman=-0.965883, AUROC=0.998882,
        highest_below_clean=0.998542
audio:  Spearman=-0.815835, AUROC=0.945113,
        highest_below_clean=0.998542
```

混淆相关：

```text
corr(q_v, vision_length)= 0.087347
corr(q_v, vision_energy)= 0.007884
corr(q_a, audio_length)= 0.252213
corr(q_a, audio_energy)=-0.024787
```

结论：视觉/音频可靠性均能稳定识别人工退化；能量捷径基本消失，但音频长度相关0.252需要跨数据集复核。

最终P4 Learned seed 1113完整686样本审计也已通过：

```text
vision: Spearman=-0.958149, AUROC=0.989878,
        highest_below_clean=0.998542
audio:  Spearman=-0.828746, AUROC=0.949643,
        highest_below_clean=0.997085
```

seed 1113混淆相关中，视觉能量为0.153522、音频长度为0.243383，后者仍需跨数据集复核。
人工严重损坏音频或视觉时，任务预测变化很小，说明MOSI上的融合模型高度依赖文本；
当前证据支持“可靠性感知的训练监督分配”，不支持“推理时动态抗噪融合”的夸大主张。

最终P4三种子完整审计汇总（均值±样本标准差）：

| Modality | Spearman(severity,q)↓ | Clean/corrupt AUROC↑ |
|---|---:|---:|
| Vision | -0.962946±0.004189 | 0.995197±0.004719 |
| Audio | -0.819850±0.007717 | 0.941968±0.009641 |

三种子的音频长度相关均值为0.241182±0.012280，是目前最稳定的混淆风险。

## 6. MOSI seed 1111正式结果

| 预算分配 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE↓ | Corr↑ | Loss↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 | 0.9567 |
| Constant | 0.8309 | 0.8300 | 0.8491 | 0.8487 | 0.5087 | 0.4504 | 0.7221 | 0.7900 | 0.9845 |
| Reversed | 0.8338 | 0.8335 | 0.8521 | 0.8524 | 0.4985 | 0.4242 | 0.7393 | 0.7993 | 1.0266 |
| Permuted | 0.8294 | 0.8284 | 0.8476 | 0.8471 | 0.5219 | 0.4577 | 0.7190 | 0.7940 | 0.9812 |
| Oracle | 0.8338 | 0.8326 | 0.8537 | 0.8531 | 0.5146 | 0.4548 | 0.7164 | 0.7923 | 0.9769 |

当前诚实结论：

- Learned相对Constant在二分类、Acc-5、MAE、Corr和Loss上更好，但Acc-7较低。
- Learned相对Reversed在除Corr外的主要指标上更好，Acc-7 +0.0204，MAE -0.0222。
- Learned相对Permuted在二分类、MAE、Corr和Loss上更好，Permuted在Acc-5/7上更好。
- Learned相对Oracle在二分类、Corr和Loss上更好；Oracle在Acc-5/7和MAE上更好。
- 单种子证据总体支持主要回归/二分类目标，但作用性证据呈现指标权衡，不支持“所有指标稳定提升”。

## 7. 当前立即任务

MOSI三种子主结果和最终可靠性审计均已冻结。P4此前只完整支持MOSI；2026-08-10已将
冻结P4按原逻辑移植到MOSEI，包括可靠性头、干预训练、干净主任务、allocation warmup、
作用性控制和日志统计。静态跨数据集一致性测试4项与Python语法编译通过；本机无PyTorch，
完整33项数值测试必须在服务器执行。服务器包：

```text
/Users/augustus/projects/论文/server-packages/mfon_mosei_p4_port_20260810.tar.gz
SHA256 9cc49c6978290e5442ac55b460c74752c2ac3fceebeb06c0232776dd330d8994
```

安装和两轮smoke步骤见 `docs/SERVER-P5-MOSEI-INSTRUCTIONS.md`。在上传前先确认
`data/MOSEI/unaligned_50.pkl`、seed 1111单模态encoder和磁盘空间；不得直接开25轮训练。

服务器预检确认MOSEI数据存在（13GB），但seeds 1111--1113均无单模态encoder。清理17个
旧MOSI诊断checkpoint后，九个最终baseline/P4 checkpoint完整保留，可用空间从1.2GB恢复到
12GB。P5包已安装；服务器33项测试全部通过，关键文件语法编译通过，MOSEI命令行限制已解除。
MOSEI seed 1111 audio encoder已经完成并确认保存：encoder 28M、decoder 1.3M；
服务器文件时间为Aug 11 00:17。seed 1111 vision encoder已于2026-08-12通过`nohup`
启动，shell报告PID 839。当前只检查`mosei_vision_encoder_1111.log`和视觉进程，
不得重复启动audio或vision；视觉完成并确认两个checkpoint后，才运行两轮融合smoke。

P4 Learned true-budget seed 1112公平测试已经完成：

| Method, seed 1112 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE↓ | Corr↑ | Loss↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired baseline | 0.8265 | 0.8258 | 0.8460 | 0.8459 | 0.4985 | 0.4490 | 0.7269 | 0.7941 | TBD |
| P4 Constant | 0.8265 | 0.8260 | 0.8476 | 0.8476 | 0.4985 | 0.4359 | 0.7295 | 0.7974 | 0.9880 |
| P4 Learned | **0.8309** | **0.8301** | **0.8506** | **0.8505** | **0.5073** | **0.4402** | **0.7187** | **0.7992** | **0.9724** |

P4 Learned在全部九项已报告指标上优于同预算P4 Constant，证明改进来自可靠性驱动的
样本分配，而不是增加总辅助损失权重。相对repaired baseline，P4 Learned在两组二分类、
Acc-5、MAE和Corr上更好，Acc-7低0.0088；不能宣称所有指标全面提升。

下一步只在seed 1113运行冻结后的P4 Learned；若继续保持baseline水平或更好，再补匹配的
P4 Constant并最终复核seed 1111。不要继续调整P4超参数，也不要把test结果用于继续调参。

P4 Learned seed 1113也已完成：Has0 Acc-2=0.8324、Has0 F1=0.8312、
Non0 Acc-2=0.8521、Non0 F1=0.8516、Acc-5=0.4942、Acc-7=0.4402、
MAE=0.7161、Corr=0.7934、Loss=0.9579。相对seed 1113 repaired baseline，
两组二分类分别提高约0.0030--0.0044，MAE降低0.0065，Corr提高0.0004；
但Acc-5和Acc-7分别下降0.0233和0.0234。主要回归/二分类趋势得到复现，
细粒度分类存在稳定性代价。下一实验是匹配的seed 1113 P4 Constant。

匹配的P4 Constant seed 1113结果为：Has0 Acc-2=0.8309、Has0 F1=0.8297、
Non0 Acc-2=0.8506、Non0 F1=0.8500、Acc-5=0.5000、Acc-7=0.4461、
MAE=0.7183、Corr=0.7924、Loss=0.9626。Learned相对Constant改善两组二分类、
MAE、Corr和Loss，但Acc-5/7分别低0.0058/0.0059。综合seeds 1112/1113，
Learned的两种子平均值除Acc-7外均优于Constant。下一步运行冻结P4 Learned seed 1111，
随后补匹配Constant，形成最终三种子公平比较。

冻结P4 Learned seed 1111结果为：Has0 Acc-2=0.8265、Has0 F1=0.8258、
Non0 Acc-2=0.8460、Non0 F1=0.8459、Acc-5=0.4956、Acc-7=0.4286、
MAE=0.7290、Corr=0.7929、Loss=0.9821。相对seed 1111 repaired baseline，
Has0 Acc持平、Has0 F1提高0.0005，其余主要指标略低（MAE高0.0011、Corr低0.0030、
Acc-7低0.0102）。该结果表明对baseline的提升尚未跨三种子稳定复现。
因此随后完成seed 1111 P4 Constant，用于判断动态分配相对等预算平均分配的三种子作用。

匹配的P4 Constant seed 1111结果为：Has0 Acc-2=0.8280、Has0 F1=0.8273、
Non0 Acc-2=0.8476、Non0 F1=0.8475、Acc-5=0.4985、Acc-7=0.4315、
MAE=0.7310、Corr=0.7913、Loss=0.9921。至此MOSI三种子公平实验完成。

三种子均值±样本标准差：

| Method | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE↓ | Corr↑ | Loss↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 0.8270±0.0009 | 0.8260±0.0008 | 0.8476±0.0016 | 0.8472±0.0014 | 0.5063±0.0099 | 0.4505±0.0125 | 0.7258±0.0028 | 0.7943±0.0015 | TBD |
| P4 Constant | 0.8285±0.0022 | 0.8277±0.0019 | 0.8486±0.0017 | 0.8484±0.0014 | 0.4990±0.0009 | 0.4378±0.0075 | 0.7263±0.0069 | 0.7937±0.0033 | 0.9809±0.0160 |
| P4 Learned | **0.8299±0.0031** | **0.8290±0.0029** | **0.8496±0.0032** | **0.8493±0.0030** | **0.4990±0.0072** | 0.4363±0.0067 | **0.7213±0.0068** | **0.7952±0.0035** | **0.9708±0.0122** |

最终MOSI结论：Learned相对Constant在两组二分类、MAE、Corr和Loss的三种子均值上更好，
Acc-5几乎相同，Acc-7低0.0015，初步支持固定预算下的可靠性分配作用。相对repaired MFON，
Learned提高二分类并改善MAE/Corr，但Acc-5低0.0073、Acc-7低0.0141；不能宣称全面提升。
仅三个种子不足以宣称统计显著，下一阶段应转向跨数据集与鲁棒性验证，而不是继续调MOSI。

### 历史诊断记录

MOSI seed 1112的Learned正式实验已完成，结果未复现seed 1111的提升：

| Method, seed 1112 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired baseline | 0.8265 | 0.8258 | 0.8460 | 0.8459 | 0.4985 | 0.4490 | 0.7269 | 0.7941 | TBD |
| Learned | 0.8163 | 0.8168 | 0.8247 | 0.8257 | 0.4636 | 0.4111 | 0.7802 | 0.7855 | 1.0828 |

Learned在全部已知指标上均差于baseline，属于实质性跨种子失败，
不应解释为普通随机波动。完整686样本可靠性审计显示：视觉
Spearman=-0.963491、AUROC=0.994929；音频Spearman=-0.789313、AUROC=0.916853。
可靠性头仍能稳定识别退化，因此当前问题定位为可靠性动态分配的优化效果不稳定。
下一个诊断实验是seed 1112 Constant；暂不运行Permuted。

seed 1112 Constant已完成：Has0 Acc-2=0.8105、Has0 F1=0.8110、Non0 Acc-2=0.8186、
Non0 F1=0.8196、Acc-5=0.4723、Acc-7=0.4155、MAE=0.7695、Corr=0.7870、
Loss=1.0607。Constant相对Learned在MAE、Corr、Loss和Acc-5/7上更好，但二分类更差；
两者均明显差于seed 1112 repaired baseline。因此动态分配不是唯一问题，
主任务特征混入人工退化是当前首要待验证原因。

已新增 `--reliability-task-corrupt-scale`。设为0时，可靠性头仍在干净/退化对上学习，
但情感主任务始终使用干净音视频。本地27项测试全部通过。下一实验是
seed 1112 Constant + clean task path，用于严格隔离主任务污染效应。

P3 Constant + clean task path正式结果为：Has0 Acc-2=0.8134、Has0 F1=0.8139、
Non0 Acc-2=0.8216、Non0 F1=0.8227、Acc-5=0.4752、Acc-7=0.4169、
MAE=0.7708、Corr=0.7869、Loss=1.0607。该结果与P2 Constant几乎一致，
说明主任务污染不是主要原因。

旧“固定预算”warmup在前10轮将辅助权重均值从0.05递增到0.5，而
repaired baseline从第1轮就使用0.5，因此并非真正的等预算对照。已新增
`--budget-warmup-mode allocation`：总预算始终为0.5，只将样本间分配从均匀逐步过渡到
可靠性感知。历史默认 `scale` 保持不变；本地29项测试通过。这是训练损失
分配路线的最后一轮修复。

P4 Constant true-budget seed 1112已完成：Has0 Acc-2=0.8265、Has0 F1=0.8260、
Non0 Acc-2=0.8476、Non0 F1=0.8476、Acc-5=0.4985、Acc-7=0.4359、
MAE=0.7295、Corr=0.7974、Loss=0.9880。相对repaired baseline，Has0 Acc-2和Acc-5
完全恢复，Non0两项和Corr略高，MAE仅差0.0026，Acc-7低0.0131。
这证明旧预算warmup是seed 1112大幅下降的主要原因。随后完成的P4 Learned公平测试
已在本节顶部记录，并在全部九项指标上优于P4 Constant。

Oracle两轮烟雾训练、checkpoint保存、重新加载和完整测试均已通过。第2轮真实日志为：

```text
q_v=0.934053, q_a=0.935742
q_v_std=0.074186, q_a_std=0.073381
w_v=w_a=w_nce_v=w_nce_a=0.1
w_v_std=0.007948, w_a_std=0.007847
```

完整测试结果为：Has0 Acc-2=0.8192、Has0 F1=0.8193、Non0 Acc-2=0.8415、
Non0 F1=0.8422、Acc-5=0.4723、Acc-7=0.3892、MAE=0.8588、Corr=0.7383、
Loss=1.3551。这些是工程烟雾结果，不与25轮正式结果作论文结论。

匹配的25轮Oracle正式对照已完成：Has0 Acc-2=0.8338、Has0 F1=0.8326、
Non0 Acc-2=0.8537、Non0 F1=0.8531、Acc-5=0.5146、Acc-7=0.4548、
MAE=0.7164、Corr=0.7923、Loss=0.9769。

已清理smoke checkpoint，可用磁盘空间恢复到6.7GB；seed 1112/1113的四个
单模态encoder checkpoint均存在。

## 8. Oracle之后的决策

1. seed 1111作用性证据呈现指标权衡，总体有利但不足以单独形成稳定结论。
2. P4 Learned与Constant三种子已完成；MOSI冻结，不再根据test结果调参。
3. 然后扩展MOSEI和SIMS，完成跨数据集、鲁棒性、消融、效率和统计显著性。
4. 小论文当前不可宣称已达到CCF-B投稿证据要求。

## 9. 关键文件

- `small-paper-draft-v1.md`：小论文草稿
- `small-paper-references.bib`：真实参考文献元数据
- `experiment-log.md`：真实实验日志
- `experiment-plan-v4.md`：实验证据计划
- `method-evolution-and-ownership-20260729.md`：方法演化和原创边界
- `HANDOFF-SERVER-EXPERIMENTS-20260731.md`：旧服务器交接，历史还原用
- `MFON/`：本地代码，不应包含数据集、checkpoint、大日志或`__pycache__`

## 10. 数据与完整性边界

- 只使用用户实际运行的数字。
- 不伪造平均值、显著性、SOTA排名或跨数据集结果。
- 不将MFON、SAM-LML、QMF等已有思想宣称为本文首创。
- 不上传MOSI/MOSEI/SIMS数据、BERT权重、服务器checkpoint、密钥或个人附件。
