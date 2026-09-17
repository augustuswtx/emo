# MOSEI实验交接（2026-09-17）

> 新Codex对话请先读本文件，再读 `PROJECT-CONTEXT-LATEST.md` 和
> `docs/experiment-log.md`。不要从零开始，不要重复启动正在运行的任务。

## 0. 2026-09-17 当前唯一有效状态

MOSEI 冻结实验已经完成，不再训练 seeds 1111/1112/1113 的 encoder、Baseline、P4
Constant 或 P4 Learned。三个种子的正式 checkpoint 均存在，干净测试、完整高斯可靠性
审计以及四类非高斯压力审计均已完成。下文保留旧运行命令仅用于复现历史，**不得按旧的
“下一项任务”描述重复启动训练**。每次重连先执行：

```bash
cd /home/jovyan/projects/MFON
pgrep -af "run_experiment.py.*MOSEI|audit_model_quality.py.*MOSEI"
df -h /home/jovyan
nvidia-smi
```

2026-09-14 的专业审稿给出 Major Revision / 5 分。2026-09-15 已完成第一轮论文与工具
大修：标题和主张收缩为“高斯退化响应分数”，Fig. 2 改为逐种子配对图，补充一般样本
重加权、课程学习与 DEAR，增加跨样本可比性/目标保真度的只读审计脚本、Inverse 接口、
C1--C5 实验矩阵、匿名代码工件构建器和逐条回应稿。**这些修改没有产生新的实验结果，
当前科学评分仍按 5/10 看待。**完整计划见：

```text
docs/major-revision-experiment-plan-20260915.md
docs/major-revision-response-20260915.md
MFON/audit_cross_sample_validity.py
```

新增 Python 文件已在本地通过 `py_compile`；服务器环境已运行 5 项新单测并全部通过。

C1 validation 审计已经完成，视觉跨样本 KL 保真度呈反向、音频关联很弱。C2 最终日程
Inverse 与 Permuted 的 seed 1111 训练和同配置重载测试也已完成；Learned 的 MAE
`0.5388` 低于 Permuted `0.5399`、Inverse `0.5405`，但 Corr `0.7742` 略低于
Inverse `0.7743`。这不足以通过多数种子同时改善两个主终点的判据，更不能改变 C1
负结果。C2 的 seeds 1112--1113 尚未运行；不得重复 seed 1111。

下一步先归档 Permuted 检查点哈希并完成不需重训的 C5 modality-utility 分层。C5
只读脚本、本地数值单测和服务器上传包已备好；尚未上传或在服务器运行。上传包为
`/Users/augustus/projects/论文/mfon_c5_modality_utility_20260917.tar.gz`，SHA256 为
`bd90cbb5af6e4b9cf080e617745fdedac6bb791301242e88bf063e5dcb2788a6`。
先做 validation 五批次小样本验证，再做三个种子的完整 test 事后探索性分层。C4
Per-sample-only 的 seed 1111 训练尚待运行，启动前必须检查后台进程、磁盘至少 12 GiB
和 GPU；同一时间只运行一个任务。当前稿件已纳入 seed 1111 C2 结果，但新版 PDF
仍需重新编译和目检。

### 冻结 MOSEI 三种子干净结果

| Method | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE↓ | Corr↑ | Loss↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 0.8131±0.0194 | 0.8183±0.0177 | 0.8579±0.0081 | 0.8576±0.0080 | 0.5513±0.0075 | 0.5347±0.0070 | 0.5312±0.0053 | 0.7746±0.0035 | 0.495199±0.005712 |
| P4 Constant | 0.8193±0.0204 | 0.8228±0.0161 | 0.8549±0.0040 | 0.8536±0.0067 | 0.5556±0.0089 | 0.5383±0.0088 | 0.5341±0.0066 | 0.7747±0.0011 | 0.500237±0.004028 |
| P4 Learned | 0.8258±0.0218 | 0.8280±0.0178 | 0.8553±0.0065 | 0.8531±0.0083 | 0.5543±0.0082 | 0.5375±0.0078 | 0.5316±0.0062 | 0.7754±0.0011 | 0.496244±0.002190 |

Learned−Constant：MAE −0.0025、Corr +0.0007、Loss −0.0040；分类指标混合。
Learned−MFON：MAE +0.0004、Corr +0.0009、Loss +0.0010。只能主张 Learned 在固定
预算下相对 Constant 的主终点均值方向有利，不能主张全面超过 MFON 或统计显著。

### 完整可靠性审计

高斯审计（full test，n=4659，三种子均值±样本标准差）：

```text
Vision Spearman=-0.931602±0.007408, AUROC=0.972161±0.009176
Audio  Spearman=-0.830330±0.005901, AUROC=0.999982±0.000014
```

未参与可靠性训练的压力扰动 AUROC：

| Corruption | Vision | Audio |
|---|---:|---:|
| timestep-dropout 0.75 | 0.498965±0.003535 | 0.593710±0.022123 |
| contiguous-mask 0.75 | 0.541152±0.011892 | 0.754388±0.009935 |
| temporal-shift 1.00 | 0.509639±0.002927 | 0.511274±0.002509 |
| modality-missing 1.00 | 0.407169±0.227165 | 0.999857±0.000248 |

结论：可靠性头稳定识别训练内高斯扰动，但不是通用质量估计器。视觉缺失响应不稳定且
平均低于随机；音频缺失可检测，但音频扰动几乎不改变任务预测。可靠性检测与任务效用
必须分开解释。扩展审计套件已在服务器通过 37/37 项测试。

### 效率审计（已完成）

2026-09-11 在单块 NVIDIA GeForce RTX 4090 D 上完成统一效率微基准。协议为 MOSEI
test 的同一 32 样本批次、seed 1111 冻结检查点、3 次预热和 20 次重复；前向+反向测量
不含数据加载、优化器更新或 checkpoint 写入。

| Method | Optimized params | Checkpoint MiB | Inference ms/batch | Samples/s | Inference peak GiB | F+B ms/batch | F+B peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 130,764,865 | 585.14 | 1040.20 | 30.76 | 1.136 | 1112.09 | 4.791 |
| P4 Constant | 130,795,779 | 585.14 | 1009.84 | 31.69 | 1.136 | 1135.09 | 4.805 |
| P4 Learned | 130,795,779 | 585.14 | 1024.97 | 31.22 | 1.136 | 1120.03 | 4.805 |

P4 新增 30,914 个参与优化的可靠性参数（相对 baseline 为 0.02364%）。Learned 相对
baseline 的前向+反向延迟高 0.71%，峰值训练显存高 0.29%；推理峰值与 checkpoint 大小
不变。一次顺序微基准中观测到的推理延迟降低 1.46% 不得写成速度提升。效率低成本门槛
已经关闭，不要为效率重复训练或重复跑 seeds 1112/1113。

### 下一步严格顺序

1. 效率审计已经完成，不再重复运行。
2. F1--F5 中英文图表已经完成并写入论文，不要重新制作同一组基础图；后续只做模板内
   尺寸和分页微调。
3. 若投稿前 GPU 预算允许，再做最终日程的 inverse/difficulty-aware 和 batch-permuted
   作用性控制；这是检验“高可靠性正向分配”而非继续刷分。
4. 官方 Springer Nature 2024-12 模板、NCA 编号引用格式和匿名稿编译预检已经完成：
   当前 A4 PDF 为 23 页，日志无 LaTeX 错误、缺失文件、未定义引用/交叉引用或 overfull；
   全页光栅化目检无裁切、重叠或缺字。Python 打包器可生成不含 `\\input` 的九文件匿名
   可编辑源码 ZIP。投稿前仍需用常规 `pdflatex`/BibTeX 或 Editorial Manager 做最终引擎复核。
5. 当前稿件和投稿状态位于 `paper/springer-nca/`，预览 PDF 位于
   `output/pdf/nca_anonymous_draft.pdf`。独立 title page 仍需作者填写姓名、单位、ORCID、
   通讯信息及各项声明；这些身份信息不得加入匿名稿或匿名 ZIP。
6. 非高斯审计代码提交为 `f341865`，效率审计工具为 `78a8ede`，旧 PyTorch 兼容修复为
   `918c41e`。

效率工具的本地上传包为
`/Users/augustus/projects/论文/mfon_efficiency_audit_20260908.tar.gz`，SHA256：
`c51063bc0b202d47905bf5a38a65479f4db7016deebf68feb7110eb957bf9bbb`。
提交 `78a8ede` 已通过 GitHub Desktop 推送，当前本地 `main` 与 `origin/main` 一致。

## 以下为历史交接记录

## 1. 项目位置

- GitHub：`https://github.com/augustuswtx/emo`
- 本地工作区：`/Users/augustus/projects/论文`
- 服务器项目：`/home/jovyan/projects/MFON`
- 当前跨数据集：CMU-MOSEI
- 数据：`/home/jovyan/projects/MFON/data/MOSEI/unaligned_50.pkl`（13GB）
- 基础模型：MFON（COLING 2025）
- 冻结候选：P4干预式可靠性学习 + 固定预算逐样本辅助监督分配

## 2. MOSI阶段已完成（探索性证据）

完整性审计确认：早期 MOSI test 可靠性诊断曾用于保留视觉头并重设计音频头。因此，
MOSI 不是未触碰的确认数据集，论文必须把其结果写成开发阶段的探索性证据。P4 冻结后
不再根据 MOSI test 调参。三种子均值±样本标准差：

| Method | Has0 Acc-2 | Non0 Acc-2 | Acc-5 | Acc-7 | MAE↓ | Corr↑ |
|---|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 0.8270±0.0009 | 0.8476±0.0016 | 0.5063±0.0099 | 0.4505±0.0125 | 0.7258±0.0028 | 0.7943±0.0015 |
| P4 Constant | 0.8285±0.0022 | 0.8486±0.0017 | 0.4990±0.0009 | 0.4378±0.0075 | 0.7263±0.0069 | 0.7937±0.0033 |
| P4 Learned | 0.8299±0.0031 | 0.8496±0.0032 | 0.4990±0.0072 | 0.4363±0.0067 | 0.7213±0.0068 | 0.7952±0.0035 |

诚实结论：Learned相对Constant改善二分类、MAE、Corr和Loss，Acc-5几乎相同，
Acc-7低0.0015；相对MFON改善二分类/MAE/Corr，但Acc-5/7下降。不能宣称全面提升。

最终P4三种子可靠性审计：

```text
Vision Spearman=-0.962946±0.004189, AUROC=0.995197±0.004719
Audio  Spearman=-0.819850±0.007717, AUROC=0.941968±0.009641
Audio-length confound=0.241182±0.012280
```

MOSI任务预测对音视频人工噪声变化很小，说明文本主导。当前只主张训练时辅助监督分配，
不主张已经实现推理时抗噪融合。

## 3. MOSEI代码和验证

P4已完整移植到MOSEI：可靠性头、ordered corruption、干净主任务通路、固定预算
allocation warmup、Learned/Constant等控制、优化器参数和日志统计。

- 代码提交：`5975209`
- 服务器验证记录：`28955a7`
- 服务器包：`mfon_mosei_p4_port_20260810.tar.gz`
- SHA256：`9cc49c6978290e5442ac55b460c74752c2ac3fceebeb06c0232776dd330d8994`
- 服务器33项测试通过（4.966秒）
- `py_compile`通过
- `run_experiment.py`已允许MOSI和MOSEI使用干预可靠性；SIMS仍未移植

## 4. 磁盘和保留证据

清理17个旧MOSI诊断checkpoint后：

```text
50GB总空间，39GB已用，12GB可用，77%占用
MOSI/save_models共5.8GB
```

每个MOSI种子1111/1112/1113仅保留：

- `baseline_pos_fixed`
- `p4_constant_true_budget`
- `p4_learned_true_budget`

不要删除MOSEI数据、BERT、MOSI单模态encoder或上述九个checkpoint。

## 5. 当前任务状态

2026-08-17，MOSEI seed 1111 的 Repaired MFON、P4 Constant 与 P4 Learned 均已完成
25 轮训练、checkpoint 保存和同配置重载测试。Constant 实验名为
`p5_mosei_p4_constant_true_budget`，正式测试结果为：
Has0 Acc-2=0.7995、Has0 F1=0.8064、Non0 Acc-2=0.8555、Non0 F1=0.8559、
Acc-5=0.5467、Acc-7=0.5304、MAE=0.5406、Corr=0.7735、
Loss=0.5008784563954897。

Learned 实验名为 `p5_mosei_p4_learned_true_budget`，正式测试结果为：Has0 Acc-2=0.8008、
Has0 F1=0.8075、Non0 Acc-2=0.8550、Non0 F1=0.8552、Acc-5=0.5463、
Acc-7=0.5302、MAE=0.5388、Corr=0.7742、Loss=0.49853695405478843。
Learned 的验证最优 checkpoint 仅在 epochs 1、2 保存，最终按冻结的验证损失规则选择
epoch 2。正式 checkpoint SHA256 为
`88f778d2d42d195d478af916d9be2400d9e2edce208d221598358d8a48d3b868`，与 smoke 的
`3a0fef1cbf039c48642678fe3625a94b0ae3fe7253b97ca4d55d8ad75006d4a6` 不同，排除误加载。

相对 Constant，Learned 的 MAE 改善 0.0018、Corr 提高 0.0007、Loss 降低约 0.002342；
二分类和细粒度分类指标有升有降。这只是 seed 1111 匹配确认性 pilot，不能形成稳定跨种子
结论。seed 1112 的音频/视觉 encoder 与 Repaired MFON 已完成。其正式 baseline 结果为：
Has0 Acc-2=0.7929、Has0 F1=0.7996、Non0 Acc-2=0.8487、Non0 F1=0.8485、
Acc-5=0.5559、Acc-7=0.5398、MAE=0.5290、Corr=0.7730、
Loss=0.49642193281591696。该单元只建立 seed 1112 参照，不形成方法结论。

seed 1112 P4 Constant 也已完成，正式结果为：Has0 Acc-2=0.8403、Has0 F1=0.8386、
Non0 Acc-2=0.8506、Non0 F1=0.8460、Acc-5=0.5645、Acc-7=0.5478、
MAE=0.5274、Corr=0.7757、Loss=0.49592633410108067。其预算合同满足
`q_v=q_a=1`、分配标准差为 0、平均辅助预算为 0.3/0.3/0.001/0.001。

下一项唯一正式任务是 seed 1112 P4 Learned；随后才进入 seed 1113。不得根据已见测试
结果改变配置、25 轮训练长度或验证选模规则。只有 Learned-minus-Constant 才隔离学习式
样本分配的贡献；Constant 与 baseline 的差异还包含可靠性训练机制。

重连后的第一组命令只能检查，不启动训练：

```bash
cd /home/jovyan/projects/MFON
pgrep -af "run_experiment.py.*MOSEI"
ls -lh MOSEI/save_models/uni_fea_encoder/MOSEI/1112/best_loss_{audio,vision}_encoder.pt
ls -lh MOSEI/save_models/all_model/MOSEI/1112/{p5_mosei_repaired_baseline,p5_mosei_p4_constant_true_budget}/TVA_fusion_model.pt
df -h /home/jovyan
nvidia-smi
```

以下 seed 1111 P4 Learned 命令已经执行完毕，仅作为可复现历史记录，不得重复启动：

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py \
  --dataset MOSEI --stage train-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 \
  --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control learned \
  --exp-name p5_mosei_p4_learned_true_budget \
  > mosei_p5_p4_learned_true_budget_1111.log 2>&1 & echo $!
```

MOSEI seed 1111 audio encoder已经完成。用户于2026-08-12确认以下两个文件存在；
服务器显示的文件修改时间为Aug 11 00:17：

```text
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_encoder.pt  28M
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_decoder.pt  1.3M
```

seed 1111 vision encoder随后已用`nohup`启动，启动时shell报告PID `839`，并已完成。
用户于2026-08-13确认以下两个文件存在；服务器显示的文件修改时间为Aug 12 05:43：

```text
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_encoder.pt  55M
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_decoder.pt  1.3M
```

以下vision启动命令仅作为历史记录，不得重复执行：

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py --dataset MOSEI --stage train-vision --seed 1111 \
  > mosei_vision_encoder_1111.log 2>&1 & echo $!
```

audio和vision前置训练、33项测试及两轮 P4 Learned smoke 均已完成。以下检查仅作为
历史记录，不得再次把 smoke 当作正式实验：

```bash
cd /home/jovyan/projects/MFON
ps -eo pid,etimes,%cpu,%mem,stat,cmd | grep "[r]un_experiment.py.*MOSEI.*train-vision.*1111"
grep -a -o "Epoch:[0-9]*" mosei_vision_encoder_1111.log | tail -n 1
tr '\r' '\n' < mosei_vision_encoder_1111.log | tail -n 20
```

视觉checkpoint历史确认命令：

```bash
ls -lh \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_encoder.pt \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_decoder.pt
```

以下audio启动命令仅作为历史记录，不得重复执行：

启动时使用的命令：

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py --dataset MOSEI --stage train-audio --seed 1111 \
  > mosei_audio_encoder_1111.log 2>&1 & echo $!
```

历史audio检查命令：

```bash
cd /home/jovyan/projects/MFON
ps -ef | grep "run_experiment.py" | grep "MOSEI" | grep "train-audio" | grep -v grep
grep -a -o "Epoch:[0-9]*" mosei_audio_encoder_1111.log | tail -n 1
tr '\r' '\n' < mosei_audio_encoder_1111.log | tail -n 15
```

完成后确认：

```bash
ls -lh \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_encoder.pt \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_decoder.pt
```

## 6. 后续严格顺序

### A. 训练seed 1111视觉encoder（已完成）

audio和vision均已完成；不要重复运行以下启动命令：

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py --dataset MOSEI --stage train-vision --seed 1111 \
  > mosei_vision_encoder_1111.log 2>&1 & echo $!
```

完成后确认`best_loss_vision_encoder.pt`和`best_loss_vision_decoder.pt`。

### B. 两轮P4 Learned smoke（训练侧已通过）

只有audio/vision encoder都存在且33项测试仍通过后运行：

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py --dataset MOSEI --stage train-fusion --seed 1111 \
  --epochs 2 --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control learned \
  --exp-name p5_mosei_p4_learned_smoke \
  > mosei_p5_p4_learned_smoke_1111.log 2>&1 & echo $!
```

MOSEI原始预算是`delta_va=0.3`、`delta_nce=0.001`。Epoch 2必须看到：

```text
w_v=w_a=0.3
w_nce_v=w_nce_a=0.001
Learned权重std非零
q_gap_v/q_gap_a为正
task_corruption_progress=0
checkpoint保存成功
```

smoke checkpoint保存后，必须使用同一配置重新加载并完成test：

```bash
cd /home/jovyan/projects/MFON
python run_experiment.py --dataset MOSEI --stage test-fusion --seed 1111 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control learned \
  --exp-name p5_mosei_p4_learned_smoke \
  2>&1 | tee mosei_p5_p4_learned_smoke_test_1111.log
```

2026-08-13用户提供的Epoch 2日志确认训练侧门槛通过：

```text
q_v=0.719222, q_a=0.943149
q_v_std=0.063758, q_a_std=0.008467
w_v=w_a=0.3
w_v_std=0.005322, w_a_std=0.000540
w_nce_v=w_nce_a=0.001
q_gap_v=0.197342, q_gap_a=0.243674
task_corruption_progress=0.0
checkpoint=/home/jovyan/projects/MFON/MOSEI/save_models/all_model/MOSEI/1111/p5_mosei_p4_learned_smoke/TVA_fusion_model.pt
```

四项均值预算匹配，Learned分配标准差非零，两个可靠性gap均为正，主任务路径保持
干净，checkpoint已保存。音频权重标准差较小但非零，需要在正式训练中继续监控，
目前不构成smoke失败。`test-fusion`重载已于2026-08-13完成，完整测试输出为：

```text
Has0 Acc-2=0.8008, Has0 F1=0.8075
Non0 Acc-2=0.8550, Non0 F1=0.8552
Acc-5=0.5463, Acc-7=0.5302
MAE=0.5388, Corr=0.7742, Loss=0.49853695405478843
```

这些数值只证明checkpoint能够重载并完成完整测试，是两轮工程smoke，不得写入论文
主结果或与25轮正式结果比较。训练、保存、重载和测试门槛现已全部通过。

### C. 正式MOSEI证据

smoke已通过。seed 1111 Repaired MFON baseline已经完成25轮训练、checkpoint重载和
完整测试，正式结果为：Has0 Acc-2=0.8150、Has0 F1=0.8205、Non0 Acc-2=0.8635、
Non0 F1=0.8634、Acc-5=0.5426、Acc-7=0.5267、MAE=0.5372、Corr=0.7721、
Loss=0.500200593969192。实验名为`p5_mosei_repaired_baseline`。

seed 1111 P4 Constant 已启动，冻结实验名为
`p5_mosei_p4_constant_true_budget`；完成训练和同配置重载测试后再运行 P4 Learned。
同一时间只启动一个。seeds 1112/1113 的匹配复现已经预先列入确认性计划，不得根据
seed 1111 test 的好坏选择性取消。任何 checkpoint 选择使用 validation；MOSEI test
输出只用于报告，不得继续调参。三种子完成后再做可靠性审计、均值/标准差和跨数据集结论。

2026-08-16 第二轮审稿期间进一步预先固定：MOSEI 以 MAE/Corr 为主要终点，
Has0/Non0 Acc-2/F1 为次要终点，Acc-5/7、Loss 与可靠性/混杂指标为诊断终点。
所有指标必须完整报告。该决定记录于 P4 Constant 正式测试结果产生前，不得在看到结果后改换主指标。

## 7. 完整性边界

- 不上传13GB数据、BERT权重、checkpoint、密钥或个人附件到GitHub。
- 不把未完成的MOSEI训练写成结果。
- 不把MFON基础结构宣称为原创。
- 不宣称SOTA、统计显著或CCF-C投稿就绪，除非后续证据真实完成。
