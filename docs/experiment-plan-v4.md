# 诊断驱动可靠多模态情感分析：实验设计 v4

> **2026-08-16 审稿后协议调整：** MOSI 测试集可靠性诊断曾影响视觉头保留与音频头重设计，因此 MOSI 统一定位为方法开发与探索性证据。P4 已在 MOSEI 正式比较前冻结；MOSEI 从现在起承担确认性验证，测试输出只用于报告，不得用于修改方法、超参数、分配方向或决定是否选择性扩展随机种子。当前服务器任务是 seed 1111 P4 Constant；只监控并等待其完成，不得重复启动。

## 当前确认性实验队列（覆盖旧的“看 seed 1111 趋势再扩展”规则）

| 顺序 | 任务 | 当前状态 | 完成条件 | 禁止事项 |
|---:|---|---|---|---|
| 1 | MOSEI seed 1111 Repaired MFON | done | 25 轮、重载、完整测试均完成 | 不重跑，不依据测试修改 P4 |
| 2 | MOSEI seed 1111 P4 Constant | running / 待服务器确认 | 25 轮完成；同配置重载；记录正式指标与预算日志 | 不重复启动，不并发启动 Learned |
| 3 | MOSEI seed 1111 P4 Learned | queued | Constant 完成后才启动；25 轮与重载测试完成 | 不依据 Constant test 改配置 |
| 4 | MOSEI seeds 1112/1113 三方法匹配复现 | precommitted | 相同配置、轮数、checkpoint 规则；报告均值±样本标准差 | 不因 seed 1111 test 方向不理想而取消或选择性报告 |
| 5 | MOSEI 可靠性与混杂审计 | planned | 三个 Learned checkpoint；Spearman、AUROC、长度/能量相关 | 不用审计结果回调已报告模型 |

若资源不足以完成 seeds 1112/1113，seed 1111 必须明确标为确认性 pilot，而不能包装成稳定跨数据集结论；后续应另设未触碰的数据来源或预注册复现实验。

## 审稿意见对应的最小补强包

1. **分配方向对照：** 在最终 P4 日程下加入等预算 `Inverse/Difficulty-aware` 控制，直接检验“高可靠性样本应获得更多辅助权重”的方向假设；Constant 仍是主匹配控制。
2. **作用性对照：** 在最终日程下至少加入 batch 内 Permuted 控制。若计算预算有限，优先级为 Inverse、Permuted、Reversed、Oracle。
3. **更贴近实际的特征退化：** 在不重新训练主模型的前提下，先对冻结 checkpoint 进行连续时间遮挡、随机时间步丢失、时间错位和整模态缺失评估。它们仍是特征级压力测试，不得表述为真实声学或视觉扰动。
4. **效率证据：** 记录参数增量、单 epoch 训练时间、完整训练时间、测试时间和峰值显存；同时说明可靠性头仅在训练阶段使用。
5. **统计报告：** 三个固定种子使用配对差值、均值±样本标准差和 bootstrap 置信区间；不因样本量小而宣称统计显著。

> **2026-07-29 实现更新**：已新增 `--use-budgeted-aux` 路径。该路径保留逐样本 KL/InfoNCE，在 batch 内固定辅助监督总预算，并记录质量与权重均值。8项CPU测试、服务器2轮GPU烟雾训练和checkpoint加载测试均已通过；两轮辅助权重均值分别为0.05和0.10，逐样本分数及权重标准差非零。旧 `--use-alw` 仅用于复现，不再作为主方法继续扩展。

## P0：固定预算逐样本辅助学习试跑

服务器目录：`/home/jovyan/projects/MFON`

先进行 2 个 epoch 的烟雾测试：

```bash
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 \
  --epochs 2 --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_smoke \
  2>&1 | tee mosi_budgeted_aux_align_smoke_1111.log
```

启动前先运行：

```bash
python -m unittest discover -s tests -v
```

烟雾测试通过后，先测试2轮checkpoint是否能正常加载，不立即运行完整25轮。`align`仍然只是未经退化单调性验证的代理；它用于验证固定预算代码路径和构成简单基线，不能直接作为论文最终可靠性分数。

P0 通过条件：

1. 8 项单元测试全部通过；
2. 每个 epoch 出现 `Budgeted auxiliary epoch` 日志；
3. MOSI 的 `delta_va=0.5`，所以 warmup 第 1 个 epoch 的 `w_v/w_a` 均值应约为 `0.05`，第 10 个 epoch 后应约为 `0.5`；
4. `w_nce_v/w_nce_a` 同理从约 `0.05` 增长至 `0.5`；
5. `q_v_std/q_a_std` 与 `w_v_std/w_a_std` 不是长期严格为 0，否则没有发生逐样本重分配；
6. 所有 loss、quality 和 weight 都是有限数值，无 `nan/inf`；
7. checkpoint 保存到 `MOSI/save_models/all_model/MOSI/1111/budgeted_aux_align_smoke/TVA_fusion_model.pt`，并能被测试阶段完整加载。

该 P0 只验证代码、预算守恒和训练稳定性，不用于宣称方法优于 baseline。完成后进入可靠性代理闸门，不批量运行 `norm/conf`，也不直接启动五种子实验。

## P0之后的决策闸门

### Gate A：代码与存储

必须同时满足：

1. 两个epoch均完成且无`nan/inf/traceback`；
2. 两轮均出现预算日志，均值分别约为0.05和0.10；
3. 质量分数和权重标准差不长期为0；
4. checkpoint完整保存并可加载；
5. 磁盘至少保留5GB，正式实验前建议保留10GB。

任一项失败时只修复该项并重跑2轮，不进入正式训练。

### Gate B：代理是否值得训练

对`align`、`conf`及后续干净—退化可靠性分数先做低成本离线审计：

1. 视觉、声学分别施加5档连续退化；
2. 检查分数与退化强度的Spearman方向；
3. 检查clean/corrupt AUROC；
4. 检查与有效长度、特征能量和标签绝对值的相关；
5. 使用常数、置换、反转和Oracle分数作为对照。

只有同时具备合理退化方向、优于随机的区分能力、不过度依赖长度/能量，并且分数变化能改变监督分配的代理，才进入25轮训练。`norm`已因噪声越大分数越高而淘汰。

### Gate C：单种子方法试验

先在MOSI seed 1111运行：

1. 修复后MFON；
2. 固定权重逐样本损失；
3. 固定预算 + align；
4. 固定预算 + 干净—退化可靠性。

只有第4项在退化曲线面积上优于第1、2项，且clean指标没有明显恶化，才扩展到五种子。若仅align有效而干预式可靠性无效，则论文不能宣称“可验证可靠性学习”，需返回方法设计。

## Mode

`design`。目标是形成面向 Neural Computing and Applications（当前按 CCF-C 目标管理）的可审计证据包；所有未完成结果均标记为 `TBD`。

## Venue and assumptions

- 当前目标期刊为 Neural Computing and Applications；实验仍按高于最低录用线的证据强度设计。
- 论文类型：诊断协议 + 轻量方法干预，而非单一 MFON 增量模块。
- 当前可用证据：MOSI 三种子 baseline/light/full、静态代码审计、CSS 代理混淆审计。

## Claim-evidence matrix

| Claim | Reviewer question | Evidence needed | 数据集/设置 | 指标 | 状态 |
|---|---|---|---|---|---|
| 常用质量代理可能不代表真实可靠性 | 这只是个别代码 bug 吗？ | 多代理、多数据集的长度混淆与退化单调性 | MOSI/MOSEI/SIMS | Spearman、clean/corrupt AUROC、混淆相关 | MOSI CSS done；其余 TBD |
| 当前 ALW 不能实现逐样本质量优化 | 数学和实现证据是什么？ | 张量形状测试、逐样本损失测试、权重梯度方向 | 单元测试 + 训练日志 | shape、梯度符号、权重轨迹 | 静态审计、8项CPU测试、2轮GPU训练和加载done |
| 新干预使质量信号真正可作用 | 分数是否只是装饰？ | 分数置换/反转/常数化，监督分配和梯度路径检查 | 三数据集，多骨干 | 权重变化、性能差、梯度敏感度 | 固定预算分配单测done；训练作用TBD |
| 新干预提高退化鲁棒性且不过度损伤 clean | 是否只对人造噪声过拟合？ | 已见与未见退化、连续强度、clean/robust trade-off | 三数据集 | 标准 MSA 指标、severity-AUC、相对下降 | TBD |
| 方法不是 MFON 特例 | 能否迁移？ | 至少三个结构不同的骨干 | MFON + 2 个公开骨干 | 同上 | TBD |

## Dataset / benchmark needs

| 数据集 | 作用 | 当前状态 | 必须保持一致的协议 |
|---|---|---|---|
| CMU-MOSI | 快速开发和消融 | 本地可用 | 官方/原代码划分、同一特征、同一 seed |
| CMU-MOSEI | 大规模英文验证 | 服务器待确认 | 与 MOSI 相同退化生成逻辑 |
| CH-SIMS | 跨语言验证 | 服务器待确认 | 保留中文标签与指标定义 |

退化族：

1. 声学加性噪声；
2. 视觉加性噪声；
3. 连续时间遮挡；
4. 随机时间步丢失；
5. 整模态缺失；
6. 时间置换或错位；
7. 两种退化组合；
8. 至少一种训练阶段未出现的未知退化。

每种退化使用固定随机种子和连续严重度网格，clean 必须作为 severity 0。退化只能基于输入生成，不能使用 test 标签选择噪声。

## Baseline matrix

| Baseline | 为什么必须有 | 公平性约束 | 可运行性 |
|---|---|---|---|
| 修复后 MFON | 直接实现基础 | 相同 encoder、特征、训练轮数和种子 | yes |
| 原 Q-DAMFON-light/full | 失败原型和机制对照 | 只用已完成真实结果；新增退化测试需同 checkpoint | partial |
| 简单固定权重/均匀融合 | 排除复杂模块带来的假增益 | 相同参数预算或报告额外参数 | yes |
| QMF | 质量感知融合代表 | 使用官方实现和相同退化协议 | unknown，需复现核查 |
| PDF | 校准动态融合代表 | 使用官方实现和相同特征/划分 | unknown，需复现核查 |
| SAM-LML | 噪声/缺失监督注意力强基线 | 采用论文公开设置并报告差异 | unknown |
| QA-MoE | ACL 2026 最近强基线 | 不以不一致特征表直接比较 | unknown |
| EBMC | CVPR 2026 最近强基线 | 优先官方代码；无法复现则明确 | unknown |
| CPSC | 表示/梯度自校准强基线 | 同一退化和 clean 评估 | unknown |

## Main experiments

1. **代理有效性审计**：`norm/align/conf/uncertainty` 在连续退化下的单调性、区分能力和长度/能量混淆。
2. **主性能**：clean MOSI/MOSEI/SIMS，五个固定种子。
3. **鲁棒性曲线**：每种退化 × 严重度，报告曲线下面积和最坏严重度。
4. **未知退化迁移**：训练不含该退化，测试时单独评估。
5. **跨骨干迁移**：同一干预接入三个骨干，不重新定义质量目标。
6. **作用性审计**：正常分数、batch 内置换、全局置换、反转、常数化、oracle 正对照。

## Ablations

| Variant | 测试的机制 | 关键观察 |
|---|---|---|
| 完整方法 | 全部约束 | TBD |
| 无质量有效性监督 | 质量是否学得到 | TBD |
| 无去混淆约束 | 长度/能量偏差是否回归 | TBD |
| 无作用性约束 | 分数是否再次变成装饰 | TBD |
| 自由正权重 | 是否出现趋零坍缩 | TBD |
| 固定预算/归一化权重 | 非坍缩约束是否必要 | TBD |
| 只建模可靠性 | 任务效用是否必须分开 | TBD |
| 只建模任务效用 | 可靠性是否必须分开 | TBD |

## Metrics

标准任务指标：Has0/Non0 Acc-2、F1、Acc-5、Acc-7、MAE、Corr。

新增机制指标：

- 质量分数与退化严重度的 Spearman 相关；
- clean/corrupt 检测 AUROC；
- 分数与长度、能量、标签强度的相关；
- 置换分数后的预测变化率与性能差；
- 辅助权重均值、标准差、分位数和趋零比例；
- 退化严重度曲线下面积与相对 clean 下降；
- 参数量、训练时间、推理时间和峰值显存。

## Statistics

- 核心确认性比较固定 seeds：`1111, 1112, 1113`；若资源允许再扩展 `1114, 1115`，但扩展决定不得依据已查看的 MOSEI test 方向；
- 主表报告均值 ± 样本标准差；
- 相同 seed、相同样本的对比使用配对统计；
- 显著性检验同时报告效应量和置信区间；
- 所有超参数只根据 validation 选择一次，禁止看 test 后调参。

## Execution priority

| Priority | 实验 | 成本 | 依赖 | Stop condition |
|---|---|---|---|---|
| P0 | MOSI CSS 长度/噪声审计 | low | 本地数据 | done |
| P0 | 原 MFON三种子复现核对 | medium | 已有checkpoint | done；五种子延后到方法pilot通过 |
| P0 | 逐样本损失和固定预算单元测试 | low | 无 GPU | 8 项测试 done |
| P0 | MOSI固定预算2轮烟雾测试 | low | 可用磁盘空间 | 训练、保存、加载和推理done |
| P1 | 质量代理离线退化审计 | low | 退化生成器 | align全量done：视觉AUROC 0.567、音频0.642；降级为弱代理对照 |
| P1 | 干预式可靠性头本地实现 | low | 固定预算路径 | done：三联退化、能量不变性和外部质量覆盖 |
| P1 | 干预式可靠性2轮GPU烟雾测试 | low | 上传P1补丁 | done；clean预警：MAE 1.1387、Corr 0.6365 |
| P1 | MOSI 1111干净—退化可靠性pilot | medium | P1 checkpoint | done（n=160）：视觉AUROC 0.981，音频0.501；视觉保留、音频重做 |
| P1.1 | 音频时序可靠性头 + 主任务10轮渐进扰动 | low | P1审计结论 | done：2轮clean闸门通过；n=686视觉/音频AUROC 0.971/0.854 |
| P2 | MOSI seed 1111 P1.1收敛训练 | medium | P1.1全量审计通过 | done：八项指标均优于匹配baseline；MAE 0.7171，Corr 0.7972；仍为单种子 |
| P2 | 25轮最佳checkpoint全量审计 | low | P1.1收敛训练 | done：视觉/音频AUROC 0.999/0.945，能量混淆近零；音频长度相关0.252 |
| P2 | 等预算作用性对照 | medium | P2正常分数路径 | constant/reversed/permuted 25轮done；learned在二分类、MAE、Corr、Loss总体更优，permuted在Acc-5/7更高；Oracle next |
| P2 | Oracle正对照 | medium | P2对照代码 | 2轮训练和保存通过，预算严格守恒；checkpoint加载TBD，随后25轮 |
| P1 | MOSI五种子 | medium | 单种子pilot通过 | 平均方向不一致则不扩展 |
| P2 | MOSEI/SIMS | high | MOSI 通过 | 两数据集均失败则收窄论文 |
| P3 | 多骨干与最近强基线 | high | 代码复现 | 无迁移性则不能声称 model-agnostic |

## No-fabrication status

除已标注的 MOSI 旧结果和 CSS 审计外，本文档没有生成任何实验结果。所有 `TBD` 必须由实际运行、匹配协议的论文报告或可核验公开结果填写。
