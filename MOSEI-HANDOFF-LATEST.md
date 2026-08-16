# MOSEI实验交接（2026-08-16）

> 新Codex对话请先读本文件，再读 `PROJECT-CONTEXT-LATEST.md` 和
> `docs/experiment-log.md`。不要从零开始，不要重复启动正在运行的任务。

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

## 5. 当前正在运行的任务

2026-08-16 当前唯一正式任务是 MOSEI seed 1111 P4 Constant，实验名：
`p5_mosei_p4_constant_true_budget`。用户最后确认启动进程为 PID 239；由于服务器会话和
网络可能变化，该 PID 只作历史记录。每次重连后必须先只读检查进程、日志和 checkpoint，
不得因为看不到原终端而重复启动。日志位于项目目录
`/home/jovyan/projects/MFON/mosei_p5_p4_constant_true_budget_1111.log`，此前从 `~` 直接
执行 `tail` 找不到日志只是路径错误，不代表训练未启动。

重连后的第一组命令只能检查，不启动训练：

```bash
cd /home/jovyan/projects/MFON
pgrep -af "run_experiment.py.*MOSEI.*train-fusion.*p5_mosei_p4_constant_true_budget"
ls -lh mosei_p5_p4_constant_true_budget_1111.log
tr '\r' '\n' < mosei_p5_p4_constant_true_budget_1111.log | tail -n 30
nvidia-smi
```

若进程存在，使用以下命令查看实时日志：

```bash
cd /home/jovyan/projects/MFON
tail -n 30 -f mosei_p5_p4_constant_true_budget_1111.log
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

audio和vision前置训练均已完成，不得重新启动。下一步先重新执行33项测试；测试通过后
才启动两轮P4 Learned smoke：

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
