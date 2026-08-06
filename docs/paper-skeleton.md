# 论文骨架（工作量扩展版）

> 目标：从单一"自适应权重"扩展为一篇有 3-4 个创新点、足够支撑顶会/硕论的完整工作
> 投稿目标：ACL / EMNLP / AAAI / ICME

---

## 论文标题

**Beyond Fixed Weights: Adaptive Optimization and Dynamic Prompting for Robust Multimodal Sentiment Analysis**

或简洁版：**DAMFON: Dynamic Adaptive Modal Feature Optimization Network for MSA**

---

## 工作量总览（给导师看的）

```
核心创新点: 4 个（比原方案多 3 个）
实验数据集: 5 个（比原方案多 2 个）  
对比方法:   15+ 个
消融实验:   8 组
可视化分析: 5 类
理论分析:   1 个完整小节
预计页数:   8-9 页（ACL/EMNLP 格式）
```

---

## Abstract

1. 多模态情感分析中长期存在模态欠优化问题——文本主导训练，视觉/听觉表示质量不足
2. 现有方法（如 MFON）通过知识蒸馏和对比学习增强弱模态，但存在三个未解决的局限：
   - 损失权重 α/β 固定，依赖暴力网格搜索
   - 可学习 prompt 对所有样本静态不变，忽略样本特异性
   - 训练策略一视同仁，缺乏从易到难的课程引导
3. 本文提出 DAMFON，包含四个核心贡献：
   - **自适应损失权重**：α/β 分解为可学习基值 × 训练进度系数 × 样本质量分数
   - **动态提示生成（DPG）**：基于样本特征动态生成模态提示，替代 MFON 的静态 prompt
   - **课程采样策略**：按样本难度排序，从易到难逐步引入高噪声样本
   - **通用性验证**：将上述机制做成即插即用模块，在 MFON、Self-MM、CubeMLP 三个架构上验证
4. 在 MOSI / MOSEI / CH-SIMS / UR-FUNNY / MUStARD 五个数据集上实验，一致且显著优于基线
5. 消融实验验证每个模块的独立贡献，理论分析从梯度方差角度解释自适应权重的有效性

---

## 1. Introduction

### 第一段：问题背景与挑战

- 多模态情感分析（MSA）从文本、视觉（面部表情）、声学（语音语调）三模态预测情感
- 核心挑战：模态信息不平衡 + 模态欠优化
  - 文本：BERT 提取的语义特征信息密度高，训练中迅速收敛
  - 视觉：面部表情受遮挡、光照、个体差异影响，有效信号稀疏
  - 声学：语调受环境噪声、说话人习惯干扰，信噪比低
- 直接拼接/注意力融合时，弱模态贡献有限，甚至引入噪声拖累文本表示

### 第二段：现有方法演进

- 早期融合：TFN（张量外积，维度爆炸）→ LMF（低秩近似）
- Transformer 时代：MulT（跨模态 Transformer）、Self-MM（自监督多任务）
- 针对模态不平衡：CubeMLP、MFON——MFON 通过 MPA + Intra-modal KD + Inter-modal CL 三条路径专门优化弱模态
- **但 MFON 的损失权重 α_v, α_a, β 是固定值**，通过搜索 100 组 (α, β) 组合选最优——三个问题：
  1. 换数据集/随机种子后需重新搜索
  2. 训练早期特征不稳定时等权蒸馏注入噪声
  3. 对所有样本一视同仁，忽略样本间质量差异

### 第三段：更深层的三个被忽视的问题

固定权重只是表象。深入分析 MFON 的设计，还存在三个同样关键但未被关注的问题：

**问题 1：静态 Prompt。** MFON 在 MPA 模块中使用可学习的 prompt 向量引导跨模态注意力，但这个 prompt 对所有输入样本完全相同。一个面部被遮挡的样本和一个表情清晰的样本，不应使用同样的 prompt —— prompt 应该根据输入特征动态调整。

**问题 2：随机采样。** 训练时样本随机打乱，高噪声样本和干净样本混在一起。模型在早期被迫拟合噪声信号，导致收敛不稳定。更合理的方式是从干净样本开始，逐步引入困难样本。

**问题 3：方法专有性。** MFON 的改进都内嵌在自身架构中，无法判断这些改进是否具有通用性。一个真正有价值的贡献应该能在不同架构上复现。

### 第四段：本文方案

- 针对问题 1 → 自适应损失权重（ALW）：三因子分解 α = αbase × s(t) × q(·)
- 针对问题 2 → 动态提示生成（DPG）：基于样本特征的小型网络生成 instance-level prompt
- 针对问题 3 → 课程采样策略（CSS）：基于质量分数对样本排序，渐进式训练
- 针对问题 4 → 即插即用泛化：将上述三个模块封装为独立组件，分别部署到 Self-MM 和 CubeMLP

### 第五段：贡献总结（4 条）

1. **自适应损失权重（ALW）**：首次将因子化自适应权重引入 MSA，含可学习基值、课程预热、质量加权三个子因子，各子因子有明确语义和独立消融
2. **动态提示生成（DPG）**：替代静态 prompt，使跨模态注意力具备样本感知能力
3. **课程采样策略（CSS）**：基于样本难度的渐进训练，提升训练稳定性和最终性能
4. **通用框架验证**：在 MFON、Self-MM、CubeMLP 三个架构上验证三个模块的即插即用性

---

## 2. Related Work

### 2.1 Multimodal Sentiment Analysis

| 方法 | 年份 | 核心思路 |
|------|------|---------|
| TFN | 2017 | 张量外积融合，维度灾难 |
| LMF | 2018 | 低秩多模态融合 |
| MulT | 2019 | 跨模态 Transformer |
| Self-MM | 2021 | 自监督多任务 + 模态专属标签 |
| CubeMLP | 2022 | MLP 混合多模态特征 |
| MFON | 2025 | MPA + KD + CL 优化弱模态 |

趋势：从融合策略 → 不平衡处理 → 弱模态专门优化

### 2.2 Knowledge Distillation in Multimodal Learning

- 传统 KD：Hinton 2015，教师-学生框架
- 多模态 KD：从单模态教师向多模态学生迁移知识
- MFON 的模态内 KD：用预训练单模态编码器监督多模态特征
- **本文差异化**：关注 KD 损失权重的动态调节，而非蒸馏架构本身

### 2.3 Adaptive & Dynamic Loss Weighting

- Uncertainty Weighting (Kendall et al., 2018)：用任务依赖不确定性作为权重
- GradNorm (Chen et al., 2018)：通过梯度范数平衡多任务学习
- Dynamic Weight Average：基于 loss 变化率调整
- MOML (Ye et al., 2023)：meta-learning 学习损失权重
- **本文差异化**：三因子分解（基值×进度×质量），每个因子有独立语义和消融

### 2.4 Prompt Learning in Vision-Language Models

- CoOp, CoCoOp：视觉 prompt learning
- MaPLe：多模态 prompt coupling
- MFON 的 MPA 中使用静态 prompt
- **本文差异化**：instance-level 动态 prompt，根据输入特征生成

### 2.5 Curriculum Learning

- Bengio et al. 2009：从易到难训练
- 在 NLP 中的应用：sentence length, word frequency 作为难度指标
- **本文差异化**：基于样本质量分数（可在线估计）的动态难度排序

---

## 3. Method

### 3.1 Preliminary: MFON 框架回顾

#### 3.1.1 特征提取

- 文本：BERT → T ∈ R^(l_t × d)
- 视觉：FACET/OpenFace → V_raw ∈ R^(l_v × d_v) → MLP → V ∈ R^(l_v × d)
- 声学：COVAREP → A_raw ∈ R^(l_a × d_a) → MLP → A ∈ R^(l_a × d)

#### 3.1.2 三个核心模块

**Modal Prompt Attention (MPA)**：

```
V' = CrossAttn(Q=T, K=V, V=V) + PromptAttn(Q=T, K=P_v, V=P_v)
A' = CrossAttn(Q=T, K=A, V=A) + PromptAttn(Q=T, K=P_a, V=P_a)
```
其中 P_v, P_a 是静态可学习 prompt 向量，对所有输入相同。

**Intra-modal Knowledge Distillation**：

```
L_kd_v = KL(σ(f_v(V_raw)/τ) || σ(V'/τ))
L_kd_a = KL(σ(f_a(A_raw)/τ) || σ(A'/τ))
```
f_v, f_a 为预训练单模态编码器（冻结）。

**Inter-modal Contrastive Learning**：

```
L_cl = L_cl(T, V') + L_cl(T, A')
```
正样本对：(T, V'), (T, A')；负样本对：batch 内其他样本。

#### 3.1.3 总损失（MFON 原文）

```
L_total = L_task + α_v·L_kd_v + α_a·L_kd_a + β·L_cl
```
α_v, α_a, β 为固定超参数，通过 100 组随机搜索确定。

### 3.2 创新 1：自适应损失权重（Adaptive Loss Weighting, ALW）

#### 3.2.1 三因子分解

以视觉模态蒸馏权重 α_v 为例：

```
α_v(t, x) = αbase_v · s(t) · q_v(x)
```

| 因子 | 符号 | 语义 | 范围 |
|------|------|------|------|
| 可学习基值 | αbase_v | 任务级：蒸馏对当前任务的基础重要程度 | R+ |
| 训练进度系数 | s(t) | 时间级：当前训练阶段适合施加多大强度的蒸馏 | [0, 1] |
| 样本质量分数 | q_v(x) | 样本级：特征质量越高，蒸馏越有效 | [0, 1] |

**因子 1：αbase — 自动学习**

```python
self.alpha_base_v = nn.Parameter(torch.tensor(0.1))  # 初始化为 MFON 最优值
self.alpha_base_a = nn.Parameter(torch.tensor(0.1))
self.beta_base = nn.Parameter(torch.tensor(0.1))
```

通过主损失梯度更新。若蒸馏有利于降低 L_task，αbase 自动增大；反之衰减。施加 softplus 保持正值。

**因子 2：s(t) — 课程预热**

```
线性: s(t) = min(t/T_warmup, 1.0)     （默认，T_warmup=10）
余弦: s(t) = 1 - cos(πt/2T_warmup)   （t ≤ T_warmup 时）
阶梯: s(t) = 0.1 · ⌈10t/T_warmup⌉   （每小段跳升）
```

默认使用线性。三种策略作为超参数消融。

为什么需要预热：前几个 epoch MPA 输出的 V' / A' 接近随机噪声，此时用 KLD 蒸馏等价于将噪声信号反向传播到编码器。预热期蒸馏权重接近 0，让 MPA 先学会基本的跨模态映射。

**因子 3：q(x) — 样本质量感知**

设计 3 种质量函数：

```
q_norm:  q_v = σ(||V'||_2 / τ_norm)
         假设：特征 L2 范数越大 → 信息越丰富

q_conf:  q_v = max_k softmax(Predictor(V'))_k
         假设：预测越自信 → 特征越可靠

q_align: q_v = (cos(V', f_v(V_raw)) + 1) / 2
         假设：与教师特征越接近 → 蒸馏效果越好
```

#### 3.2.2 完整的自适应损失

```
L_total = L_task
        + α_v(t, x) · L_kd_v
        + α_a(t, x) · L_kd_a
        + β(t, x) · L_cl

α_v(t, x) = αbase_v · s(t) · q_v(x)
α_a(t, x) = αbase_a · s(t) · q_a(x)
β(t, x)  = βbase · s(t) · (q_v(x) + q_a(x)) / 2
```

### 3.3 创新 2：动态提示生成（Dynamic Prompt Generation, DPG）

#### 3.3.1 动机

MFON 的 MPA 中，prompt P_v, P_a 在所有样本上完全相同。

两个样本：
- 样本 A：正面人脸、光照良好、表情夸张（视觉信息丰富）
- 样本 B：侧脸遮挡、光照不足、表情微弱（视觉信息稀疏）

相同的 prompt 无法同时适应这两种情况。应该为每个样本生成专属的 prompt。

#### 3.3.2 设计

```
P_v(x) = MLP_prompt_v(Concat(V, T_pooled))   # 基于样本特征动态生成
P_a(x) = MLP_prompt_a(Concat(A, T_pooled))

V' = CrossAttn(Q=T, K=V, V=V)
   + CrossAttn(Q=T, K=P_v(x), V=P_v(x))       # ← 动态 prompt
```

其中：
- `MLP_prompt_v/a`：2 层 MLP + LayerNorm + GELU，输入维度 d+d，输出 m×d（m 为 prompt 长度，d 为隐层维度）
- 计算开销：每个样本多 2 次小 MLP 前向，参数量约 m×d×2×2 ≈ 可忽略
- T_pooled 为文本的 mean pooling 表示，提供全局语义上下文

#### 3.3.3 与 ALW 的协同

DPG 改善特征 V'/A' 的质量 → q(x) 上升 → 蒸馏权重自动增大 → 蒸馏更有效。两个模块形成正向反馈。

### 3.4 创新 3：课程采样策略（Curriculum Sampling Strategy, CSS）

#### 3.4.1 动机

标准训练：样本随机 shuffle，每个 epoch 中干净样本和噪声样本均匀混合。

问题：训练初期模型能力弱，被噪声样本主导梯度，收敛路径曲折。

#### 3.4.2 设计

```
Step 1: 每个 epoch 结束时，为所有样本计算质量分数 q_i = (q_v_i + q_a_i) / 2
Step 2: 按 q_i 排序，从小到大对应"容易→困难"
Step 3: 下一个 epoch 开始时，按当前课程进度 c(e) 选择样本子集
        e: 当前 epoch
        c(e) = min(e/E_curriculum, 1.0)  # E_curriculum = 20
        epoch e 使用的样本 = {样本 i | q_i ≤ percentile(Q, c(e)×100)}
        即 epoch 1 只用最易的 5% → epoch 20 用全部样本
Step 4: 在选中的样本子集内随机 shuffle
```

#### 3.4.3 与 ALW 的协同

ALW 的 s(t) 在**损失权重**层面做预热，CSS 在**数据**层面做预热。两者从不同维度保护早期训练。

CSS 的样本排序依赖 ALW 的 q(x)，两个模块共享质量估计。

### 3.5 创新 4：通用框架验证

将 ALW、DPG、CSS 三个模块设计为即插即用组件：

| 组件 | 输入 | 输出 | 对主架构的侵入 |
|------|------|------|---------------|
| ALW | V', A', f_v, f_a | 动态权重 α, β | 仅修改 loss 计算 |
| DPG | V, A, T_pooled | 动态 prompt P(x) | 替换 MPA 中静态 P |
| CSS | q_v, q_a | 样本子集 | 仅修改 data loader |

部署到 3 个架构：

| 架构 | 需添加/替换的部分 |
|------|------------------|
| MFON | native（原生实现） |
| Self-MM | 在其多任务损失中引入 ALW；在其跨模态注意力中引入 DPG |
| CubeMLP | 在其 MLP 融合层后添加 DPG 引导的跨模态门控 |

通用性验证的意义：证明这不是一个仅对 MFON 有效的特化改进，而是一个通用的 MSA 优化框架。

### 3.6 DAMFON 总体架构

```
┌─────────────────────────────────────────────────────┐
│                    Input Sample x                    │
│  Text (BERT)    Visual (FACET)    Acoustic (COVAREP)│
│       │              │                   │          │
│       ▼              ▼                   ▼          │
│    T ∈ R^d        V_raw              A_raw          │
│       │              │                   │          │
│       │         ┌────▼────┐        ┌────▼────┐     │
│       │         │  MLP    │        │  MLP    │     │
│       │         └────┬────┘        └────┬────┘     │
│       │              ▼                   ▼          │
│       │           V ∈ R^d            A ∈ R^d        │
│       │              │                   │          │
│       ├──────────────┼───────────────────┤          │
│       │         ┌────▼────┐        ┌────▼────┐     │
│       │         │  DPG    │        │  DPG    │     │
│       │         │ P_v(x)  │        │ P_a(x)  │     │
│       │         └────┬────┘        └────┬────┘     │
│       │              │                   │          │
│       │    ┌─────────▼─────────┐ ┌──────▼────────┐ │
│       │    │  MPA (Text→Vis)  │ │ MPA (Text→Aud)│ │
│       │    └─────────┬─────────┘ └──────┬────────┘ │
│       │              ▼                   ▼          │
│       │           V' ∈ R^d           A' ∈ R^d       │
│       │              │                   │          │
│       │    ┌─────────▼─────────┐ ┌──────▼────────┐ │
│       │    │ Intra-modal KD   │ │ Intra-modal KD│ │
│       │    │  L_kd_v          │ │  L_kd_a       │ │
│       │    │  f_v(V_raw)      │ │  f_a(A_raw)   │ │
│       │    └─────────┬─────────┘ └──────┬────────┘ │
│       │              │                   │          │
│       │              │   ┌───────────────▼────────┐ │
│       │              │   │ Inter-modal CL         │ │
│       │              │   │  L_cl(T,V') L_cl(T,A')│ │
│       │              │   └───────────────┬────────┘ │
│       │              │                   │          │
│       ├──────────────┴───────────────────┤          │
│       │         ALW: α_v(t,x)  α_a(t,x)  β(t,x)   │
│       │              │                   │          │
│       │    ┌─────────▼───────────────────▼────────┐ │
│       │    │     Fusion + Predictor               │ │
│       │    │     L_task                           │ │
│       │    └──────────────────────────────────────┘ │
│       │                                             │
│       │    CSS: 按 q(x) 排序 → 课程采样             │
│       └──────────────────────────────────────────────│
└─────────────────────────────────────────────────────┘
```

### 3.7 MFON vs DAMFON 完整对照

| 维度 | MFON (COLING 2025) | DAMFON (本文) |
|------|-------------------|-------------|
| 跨模态注意力 | MPA + 静态 prompt | MPA + 动态 prompt (DPG) |
| 蒸馏损失权重 α | 固定，网格搜索 | 自适应 ALW |
| 对比损失权重 β | 固定，网格搜索 | 自适应 ALW |
| 训练策略 | 随机采样 | 课程采样 (CSS) |
| 理论分析 | 无 | 梯度方差分析 |
| 参数量（新增） | — | ~2K（可忽略） |
| 推理开销（新增） | — | 0（权重+prompt不参与推理） |
| 通用性验证 | — | Self-MM + CubeMLP |
| 数据集 | MOSI, MOSEI(, CH-SIMS) | + UR-FUNNY + MUStARD |

---

## 4. Theoretical Analysis（可选但强烈推荐，大幅增加工作量）

### 4.1 固定权重的梯度方差问题

对于固定权重 α，蒸馏损失的梯度为：

```
g_kd = α · ∇_θ L_kd
```

在训练早期（t 较小），MPA 输出的 V' 不稳定，Var[∇_θ L_kd] 很大。此时等权累加到主任务梯度上，导致总梯度方差增大：

```
Var[g_total] = Var[g_task] + α² · Var[g_kd] + Cov(g_task, g_kd)
```

当 Var[g_kd] 大且 α 固定时，总梯度方差被放大。

### 4.2 自适应权重的方差控制

本文的 α(t, x) = αbase · s(t) · q(x)，其中：
- 当 t 小时，s(t) → 0.01，整体梯度被缩小 100 倍，Var 贡献降至 1/10000
- 当 q(x) 小时（低质量样本），梯度被进一步缩小

因此自适应权重等价于在训练早期对梯度做 variance clipping，有效控制训练稳定性。

### 4.3 命题（可放入论文）

**命题 1**：存在一个 epoch t_0，使得 t < t_0 时，自适应权重的梯度方差严格小于固定权重。

**命题 2**：若 q(x) 是特征质量的单调递增函数，则自适应权重下的梯度估计具有更小的 excess variance。

（具体证明需要配合实验，可作为理论-实验结合的小节）

---

## 5. Experiments

### 5.1 实验设置

**数据集（5 个）**：

| 数据集 | 样本数 | 模态 | 任务 | 来源 |
|--------|-------|------|------|------|
| MOSI | 2199 | T+V+A | 二分类/七分类/回归 | 英文YouTube |
| MOSEI | 22856 | T+V+A | 同上 | 英文YouTube |
| CH-SIMS | 2281 | T+V+A | 同上 | 中文电影 |
| UR-FUNNY | 16514 | T+V+A | 二分类（幽默检测） | 英文TED |
| MUStARD | 690 | T+V+A | 二分类（讽刺检测） | 英文情景剧 |

新增 UR-FUNNY 和 MUStARD 的原因：
- 均为公开 MSA 基准数据集
- 幽默/讽刺检测比通用情感分析更具挑战，需要更强的模态间推理
- 验证方法在跨域数据上的泛化性

**评价指标**：
- 分类：Acc-2, Acc-7, F1
- 回归：MAE, Corr

**Baselines（15+ 个）**：

| 类别 | 方法 |
|------|------|
| 早期融合 | TFN, LMF, MFM |
| Transformer | MulT, PMR |
| 自监督/多任务 | Self-MM, SENTIME, M3SA |
| MLP 系 | CubeMLP |
| 模态优化 | MFON（直接对标） |
| 本文 | DAMFON (q_norm), DAMFON (q_conf), DAMFON (q_align) |

**实现细节**：
- BERT-base-uncased, batch size 64, max_seq_len 128
- AdamW, lr_backbone=1e-4, lr_alpha_beta=1e-3
- T_warmup=10, E_curriculum=20, τ=1.0
- 预训练编码器：f_v (ViT-B), f_a (HuBERT)
- 所有实验 5 个随机种子，报告均值±标准差

### 5.2 主实验结果

#### 5.2.1 MOSI

（此处放入完整的数据表，包括 Acc-2, Acc-7, F1, MAE, Corr）
（复制 MFON 原论文的 baselines 数字 + 新增 DAMFON 三行）

#### 5.2.2 MOSEI

（同上格式）

#### 5.2.3 CH-SIMS

（同上格式）

#### 5.2.4 UR-FUNNY（跨域泛化）

| 方法 | Acc-2 | F1 | MAE |
|------|-------|----|-----|
| MulT | | | |
| Self-MM | | | |
| CubeMLP | | | |
| MFON | | | |
| **DAMFON (best)** | | | |

#### 5.2.5 MUStARD（挑战性任务）

（同上格式）

**预期结论**：DAMFON 在所有数据集上一致优于 MFON，在 UR-FUNNY 和 MUStARD 上的优势应更明显（因为这两个数据集的模态质量差异更大，自适应策略的优势更突出）。

### 5.3 消融实验（8 组）

| 配置 | ALW | DPG | CSS | MOSI Acc-2 | MOSEI Acc-2 |
|------|:---:|:---:|:---:|------------|-------------|
| A: MFON 复现 | ✗ | ✗ | ✗ | (baseline) | (baseline) |
| B: +ALW only | ✓ | ✗ | ✗ | | |
| C: +DPG only | ✗ | ✓ | ✗ | | |
| D: +CSS only | ✗ | ✗ | ✓ | | |
| E: +ALW+DPG | ✓ | ✓ | ✗ | | |
| F: +ALW+CSS | ✓ | ✗ | ✓ | | |
| G: +DPG+CSS | ✗ | ✓ | ✓ | | |
| H: DAMFON full | ✓ | ✓ | ✓ | (best) | (best) |

含义：
- 每个模块独立有效（A vs B/C/D）
- 两两组合有正向交互（E/F/G vs B/C/D）
- 三模块全开最优（H vs E/F/G）

### 5.4 ALW 因子消融

| 配置 | αbase | s(t) | q(·) | MOSI Acc-2 | MOSI F1 |
|------|:-----:|:----:|:----:|------------|---------|
| 固定权重 (MFON) | ✗ | ✗ | ✗ | | |
| 仅可学习基值 | ✓ | ✗ | ✗ | | |
| 仅预热调度 | ✗ | ✓ | ✗ | | |
| 仅质量加权 | ✗ | ✗ | ✓ | | |
| base + s(t) | ✓ | ✓ | ✗ | | |
| base + q | ✓ | ✗ | ✓ | | |
| s(t) + q | ✗ | ✓ | ✓ | | |
| 完整 ALW | ✓ | ✓ | ✓ | | |

结论：每个因子独立有用，三者组合最优。预热因子在训练早期贡献大于后期。

### 5.5 质量函数对比

| 质量函数 | MOSI | MOSEI | CH-SIMS | UR-FUNNY |
|----------|------|-------|---------|----------|
| q_norm | | | | |
| q_conf | | | | |
| q_align | | | | |
| q_norm + q_conf | | | | |
| 三者融合 | | | | |

分析：哪个质量函数最有效？不同数据集上是否有差异？为什么？

### 5.6 DPG 分析

- 静态 prompt vs 动态 prompt 的 t-SNE 可视化对比
- 不同质量样本的 prompt 差异度（cosine distance between P_v(x_i) and P_v(x_j)）
- Prompt 长度 m ∈ {4, 8, 16, 32} 的灵敏度

### 5.7 CSS 分析

- 不同课程长度 E_curriculum ∈ {10, 20, 30, 40} 的影响
- 课程 vs 随机采样的训练 loss 曲线对比（含方差带）
- 课程采样对最终模型的样本利用率

### 5.8 通用性验证

| 架构 | 原始 | +ALW | +DPG | +CSS | +All Three |
|------|------|------|------|------|------------|
| MFON | (base) | +Δ | +Δ | +Δ | +Δ_max |
| Self-MM | (base) | +Δ | +Δ | +Δ | +Δ_max |
| CubeMLP | (base) | +Δ | +Δ | +Δ | +Δ_max |

结论：三个模块在三个架构上均有一致提升，说明本文方法是通用框架而非特化改进。

### 5.9 训练动态可视化

- α_v(t), α_a(t), β(t) 随 epoch 变化曲线
- 质量分数 q(x) 的分布直方图（按 epoch 分面）
- 梯度范数 ||g||_2 随训练的变化（固定 vs 自适应）
- t-SNE 特征可视化（MFON vs DAMFON，三模态分别）

### 5.10 超参数灵敏度

- T_warmup ∈ {5, 10, 15, 20}
- αbase_init ∈ {0.01, 0.05, 0.1, 0.5}
- 预热策略：线性 vs 余弦 vs 阶梯

预期：方法对超参数不敏感，在宽范围内均优于 MFON 固定权重。

### 5.11 噪声鲁棒性实验

向视觉/听觉模态注入不同程度的高斯噪声（σ ∈ {0, 0.1, 0.3, 0.5}）：

| 噪声强度 σ | MFON Acc-2 | DAMFON Acc-2 | 相对提升 |
|------------|-----------|-------------|---------|
| 0 (clean) | | | |
| 0.1 | | | |
| 0.3 | | | |
| 0.5 | | | |

预期：噪声越大，DAMFON 的相对优势越显著（因为质量加权会自动压低高噪声样本的权重）。

---

## 6. Results Analysis & Discussion

### 6.1 各模块贡献度分析

用 Shapley 值或简单的增量收益分解，量化 ALW/DPG/CSS 各贡献多少。

### 6.2 计算开销分析

| 指标 | MFON | DAMFON |
|------|------|--------|
| 可训练参数（新增） | — | ~4K |
| 训练时间（相对） | 1.0× | ~1.05× |
| 推理时间（相对） | 1.0× | 1.0× |
| GPU 显存（相对） | 1.0× | ~1.02× |

### 6.3 局限性

- DPG 生成的 prompt 缺乏可解释性
- q(x) 三种设计均为启发式，更系统的质量定义值得探索
- 仅验证了多模态情感分析任务，VQA/跨模态检索等留待未来

---

## 7. Conclusion

本文提出了 DAMFON——一个包含三个互补模块的 MSA 优化框架：
1. ALW 解决损失权重的静态性问题
2. DPG 解决 prompt 的静态性问题
3. CSS 解决采样的静态性问题

三个模块共享统一的质量估计，形成正向反馈闭环。在 5 个数据集、3 个架构、15+ 个基线上一致验证有效性。推理阶段零额外开销。

---

## 8. 时间规划（12 周版，适合硕士论文节奏）

| 周次 | 任务 |
|------|------|
| 1 | 复现 MFON，跑通 MOSI/MOSEI/CH-SIMS |
| 2 | 实现 ALW（αbase + s(t) + q_norm/q_conf/q_align） |
| 3 | ALW 消融实验（8 组因子消融） |
| 4 | 实现 DPG 模块，Debug，初步实验 |
| 5 | 实现 CSS 模块，Debug，初步实验 |
| 6 | DAMFON 完整版训练 + 主实验结果 |
| 7 | UR-FUNNY + MUStARD 数据集准备与实验 |
| 8 | 通用性验证（Self-MM + CubeMLP 复现 + 部署） |
| 9 | 可视化（t-SNE, 权重曲线, 梯度范数, 噪声鲁棒性） |
| 10 | 写论文初稿（用 awesome-ai-research-writing prompt 流水线） |
| 11 | 导师/同门反馈 → 修改 → 去AI味 |
| 12 | 终稿打磨 + 格式检查 + 投稿/论文提交 |

---

## 9. 配套工具使用指南

### Prompt 使用计划（awesome-ai-research-writing-prompts.md）

| 论文章节 | 使用的 Prompt | 流程 |
|---------|-------------|------|
| Abstract | 中转英 → 缩写 → 表达润色 → 去AI味 → 逻辑检查 | 中文草稿 → 4 步流水线 |
| Introduction | 中转英 → 扩写 → 表达润色 → 逻辑检查 | 同上 |
| Related Work | 英转中（读论文用）→ 中转英（写段落用） | 读→总结→写 |
| Method | 中转英 + 论文架构图 | 中文逻辑 → 英文 + 框架图 |
| Experiments | 实验分析 + 图标题 + 表标题 + 实验绘图推荐 | 数据→文本→图表 |
| 全文 | Reviewer 审视 + 去AI味 + 逻辑检查 | 终审三件套 |

### 验证里程碑

| 节点 | 验证方式 | by whom |
|------|---------|---------|
| 方法设计 | 与导师讨论 4 个贡献点 | 你自己 |
| 基线复现 | MFON 代码跑通，数值与原文一致 | 你自己 |
| 初步结果 | ALW 在 MOSI 上 p<0.05 优于 MFON | 你自己 |
| 论文骨架 | 用本文档与导师对齐 | 导师 |
| 完整实验 | 5 数据集 + 3 架构 | 你自己 |
| 论文初稿 | 用 prompt 流水线生成 | 你自己 |
| 论文终稿 | Reviewer 审视 → 同学互审 → 导师终审 | 导师+同门 |
