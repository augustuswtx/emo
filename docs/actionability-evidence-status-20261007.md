# 行动性审计：现有证据与待补文件

日期：2026-10-07。研究对象仍是同一篇小论文；当前目标是审计多模态情感分析中，模态分数能否支持逐样本辅助训练权重。此文件是实验状态记录，不是论文结论或新实验结果。

2026-10-09 更新：大论文已改为整合两篇小论文；本文件记录的是**第一篇**的历史证据状态。当前投稿路线见[双论文与 10 月冲刺计划](two-paper-thesis-and-october-2026-submission.md)。

## 已在本地核对的来源

[统一证据清单](actionability-evidence-manifest-20261007.json)由只读脚本生成，包含 130 条指标记录、来源文件路径与 SHA-256、数据集、划分、种子、分数、模态和分析状态。每个来源的证据等级如下。

| 来源 | 本地文件类型 | 当前能够说明的事 | 边界 |
|---|---|---|---|
| F3/F4 | 三种子派生汇总表 | Learned 分数识别受控退化，记录跨扰动任务影响 | 只有汇总均值和标准差；不是逐样本原始记录 |
| C1 | 三种子派生汇总表 | MOSEI valid 上视觉分数与 `-KL` 的 Spearman 平均为 -0.315；音频相关较弱 | 事后探索；InfoNCE 仅在 batch 内比较 |
| C2 | seed 1111 四组测试指标表 | Learned MAE 0.5388，Constant 0.5406，Permuted 0.5399，Inverse 0.5405；Inverse Corr 0.7743，Learned 0.7742 | 数值来自整理表；训练与测试原始日志、部分 checkpoint 哈希尚未归档到本地；只有一个种子 |
| C5 | seed 1111 事后零化效用分层 JSON | 记录视觉、音频三个分层的 Learned−Constant 指标差值 | 测试集已查看；零化是分层代理，不能证明因果模态贡献 |
| 主实验 | 三种子派生结果表 | 现有模型与 Constant/Learned 的最终性能比较 | 不是逐样本预测，且提升很小、不全面 |

P6/P7 的数值已由服务器终端输出记录在[实验日志](experiment-log.md)，但其原始文件不在本地。因此本版机器可检验清单将 `p6`、`p6_records`、`p7`、`p7_records` 标为 **missing**，没有把终端转录当作独立复核。P6 的 `all` 模型 AUROC 0.510、Spearman 0.013；P7 的重复情境符号一致率 0.496、固定 MLP AUROC 0.494。这些结果仅支持“所测试的最终 checkpoint、视觉参数块与单种子设置下，样本内在效用预测未通过门槛”，不能推出所有分数或模型都无效。

## 当前科研判断

现有结果表明：**检测退化**与**判断辅助目标保真度**、**预测局部更新效用**、**证明正确分配有收益**是四个不同的问题。F3/F4 支持第一个问题；C1 对视觉目标保真度给出反向证据；P6/P7 的终端结果不支持现有轻量效用预测器；C2 单种子性能差异太小，尚不足以证明分配机制。这一“分数到训练行动的证据断层”可以成为审计型论文的问题定义，但还不是覆盖多模型多数据集的已完成贡献。

## 归档与下一步

1. 从服务器保存下列四个 JSON；保留原名：
   - `visual_utility_seed1111_final_64pairs_20260928.json`（P6 逐样本标签）
   - `visual_utility_predictor_seed1111_final_20260928.json`（P6 分析）
   - `visual_utility_stability_seed1111_final_24groups_20261003.json`（P7 重复情境标签）
   - `visual_utility_stability_analysis_seed1111_final_20261003.json`（P7 分析）
2. 归档 C2 四组原始训练/测试日志及 checkpoint 哈希，核对整理表的每个数值。当前 `c2_primary` 仍标为缺失。
3. 补齐来源后重新生成清单，核对 P6/P7 原始 JSON 与汇总 JSON 中的文件名、种子、checkpoint 和样本数。脚本只读取输入，不修改任何实验文件。
4. 再冻结 MFON/MOSEI 的五类分数定义与分组规则，开展各分数的只读四层审计；不恢复已被 P6/P7 闸门否决的样本效用预测器训练。

在仓库根目录运行的重建命令如下；新输出名必须不存在：

```bash
python MFON/build_actionability_evidence_manifest.py \
  --f3 paper/figures/data/f3_mosei_gaussian_audit.csv \
  --f4 paper/figures/data/f4_mosei_cross_corruption.csv \
  --c1 paper/figures/data/c1_mosei_cross_sample_validity.csv \
  --c2 paper/figures/data/mosei_final_schedule_allocation_seed1111.csv \
  --c5 paper/figures/data/mosei_c5_zero_ablation_seed1111.json \
  --main paper/figures/data/f2_mosei_seed_results.csv \
  --output docs/actionability-evidence-manifest-rebuilt.json
```

取得 P6/P7 文件后，在上面的命令中加入 `--p6`、`--p6-records`、`--p7`、`--p7-records` 及相应文件路径。清单会记录文件哈希，并验证分析文件引用的原始文件名、种子、checkpoint 及样本数。原始 JSON 仍应单独保存；清单只提取指标，不复制逐样本记录。

若四个文件还在服务器的 `~/projects/MFON`，可复制以下**整行命令**将它们打成一个可下载归档。命令会检查文件存在和非空，且拒绝覆盖同名归档；它不会启动实验。

```bash
bash -c 'set -e; cd ~/projects/MFON; out=actionability_p6_p7_raw_20261007.tar.gz; test ! -e "$out" || { echo "归档已存在：$out"; exit 1; }; for f in visual_utility_seed1111_final_64pairs_20260928.json visual_utility_predictor_seed1111_final_20260928.json visual_utility_stability_seed1111_final_24groups_20261003.json visual_utility_stability_analysis_seed1111_final_20261003.json; do test -s "$f" || { echo "缺少或为空：$f"; exit 1; }; done; tar -czf "$out" visual_utility_seed1111_final_64pairs_20260928.json visual_utility_predictor_seed1111_final_20260928.json visual_utility_stability_seed1111_final_24groups_20261003.json visual_utility_stability_analysis_seed1111_final_20261003.json; tar -tzf "$out" >/dev/null; sha256sum "$out"; ls -lh "$out"'
```
