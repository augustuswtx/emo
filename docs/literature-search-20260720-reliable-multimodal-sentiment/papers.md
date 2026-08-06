# 可靠多模态情感分析文献地图（检索日期：2026-07-20）

## 1. 检索结论

当前“质量估计 + 动态加权/路由 + 缺失或噪声鲁棒性”已经是高度拥挤的方向。QMF、PDF、QA-MoE、EBMC、SAM-LML、CPSC、MRUF 等工作已经覆盖了不确定性估计、逐样本质量、专家路由、监督注意力、反事实贡献和连续退化谱。原 Q-DAMFON 的 `ALW + DPG + CSS` 不能再以“首次质量感知动态融合”为核心创新。

更有价值的研究问题是：**质量分数是否有效、是否真正影响模型决策或优化，以及是否被长度、模态能量和批次聚合等因素混淆。** 这条路线需要“诊断协议 + 机制修复 + 跨模型证据”，而不是继续堆模块。

评分说明：`洞见/完整度/数值证据` 为本项目筛选优先级（1--5），不是对论文质量或录用概率的评分。`A` 表示应精读或作为主基线，`B` 表示用于补足背景，`Risk` 表示与当前想法高度重叠。

## 2. 核心论文

| 论文 | 发表信息 | 与本项目的关系 | 洞见 | 完整度 | 数值证据 | 优先级 |
|---|---|---|---:|---:|---:|---|
| [Provable Dynamic Fusion for Low-Quality Multimodal Data](https://proceedings.mlr.press/v202/zhang23ar.html) | ICML 2023 | QMF；质量感知动态融合和理论基础 | 5 | 4 | 4 | A/Risk |
| [Predictive Dynamic Fusion](https://proceedings.mlr.press/v235/cao24c.html) | ICML 2024 | Mono/Holo confidence、相对校准和动态融合理论 | 5 | 4 | 4 | A/Risk |
| [Enhancing Multimodal Cooperation via Sample-level Modality Valuation](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.html) | CVPR 2024 | 逐样本模态贡献估计与弱模态定向优化 | 5 | 4 | 4 | A/Risk |
| [Analyzing Modality Robustness in Multimodal Sentiment Analysis](https://aclanthology.org/2022.naacl-main.50/) | NAACL 2022 | MOSI/MOSEI 鲁棒性诊断和稳健训练基准 | 5 | 5 | 5 | A |
| [Modal Feature Optimization Network with Prompt for Multimodal Sentiment Analysis](https://aclanthology.org/2025.coling-main.309/) | COLING 2025 | 当前代码和论文的直接实现基础 | 4 | 4 | 4 | A |
| [Supervised Attention Mechanism for Low-quality Multimodal Data](https://aclanthology.org/2025.emnlp-main.1084.pdf) | EMNLP 2025 | 统一噪声/缺失，显式监督注意力并使用排序约束 | 5 | 5 | 4 | A/Risk |
| [Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data](https://aclanthology.org/2025.acl-long.1075/) | ACL 2025 | 高斯潜变量不确定性、代理模态和动态注入 | 4 | 4 | 4 | A |
| [DFMU: Distribution-based Framework for Modeling Aleatoric Uncertainty in Multimodal Sentiment Analysis](https://www.ijcai.org/proceedings/2025/917) | IJCAI 2025 | 区分标注主观性和特征集合关系导致的不确定性 | 5 | 4 | 4 | A |
| [QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware Mixture of Experts](https://aclanthology.org/2026.acl-long.1461/) | ACL 2026 | 连续可靠性谱、自监督偶然不确定性和质量路由 | 5 | 5 | 5 | A/Risk |
| [Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html) | CVPR 2026 | 能量协调、逐样本信任蒸馏和鲁棒融合 | 5 | 5 | 5 | A/Risk |
| [Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Multimodal_Learning_on_Low-Quality_Data_with_Conformal_Predictive_Self-Calibration_CVPR_2026_paper.html) | CVPR 2026 | 可靠性驱动的表示和梯度自校准 | 5 | 5 | 5 | A/Risk |
| [TMDC: A Two-Stage Modality Denoising and Complementation Framework](https://ojs.aaai.org/index.php/AAAI/article/view/37212) | AAAI 2026 | 噪声与缺失模态的两阶段去噪/补全 | 4 | 4 | 4 | A |
| [Recovering Coherent Affective Patterns](https://ojs.aaai.org/index.php/AAAI/article/view/39349) | AAAI 2026 | 缺失模态下的时序结构恢复和自适应融合 | 4 | 4 | 4 | A |
| [Uncertainty-Calibrated Elastic Alignment](https://aclanthology.org/2026.findings-acl.260/) | Findings ACL 2026 | 概率补全、不确定性驱动弹性对齐 | 4 | 4 | 4 | A |
| [DEAR: Distributional Error-Aware Reliability](https://aclanthology.org/2026.findings-acl.1517/) | Findings ACL 2026 | 重构误差、可靠性门控和缺失模态风险 | 4 | 4 | 4 | A/Risk |
| [When Does Quality-Aware Multimodal Fusion Matter?](https://arxiv.org/abs/2606.26473) | INTERSPEECH 2026；arXiv 页面 | 用分数置换检查可靠性是否真正影响决策 | 5 | 4 | 4 | A/Risk |
| [MRUF: Multi-granularity Routing with Uncertainty-Aware Fusion](https://arxiv.org/abs/2607.10599) | 2026 arXiv 预印本 | 留一误差监督模态重要性，多粒度路由和不确定性校准 | 4 | 3 | 3 | Risk |
| [MoLAN: A Unified Modality-Aware Noise Dynamic Editing Framework](https://aclanthology.org/2026.findings-acl.1225/) | Findings ACL 2026 | 细粒度分块去噪，说明“局部质量”也已被研究 | 4 | 4 | 4 | Risk |

## 3. 研究簇与机会图

### 簇 A：可靠性估计与动态融合

代表：QMF、PDF、QA-MoE、EBMC、MRUF、DEAR。

结论：中心主张已经被覆盖。仅用范数、置信度或对齐度生成权重，再做门控或 MoE，难以形成新颖贡献。

仍可研究：质量代理是否有效、是否被混淆、是否真正作用于决策/梯度，以及在未知退化下是否仍可校准。

### 簇 B：噪声、缺失和鲁棒性

代表：NAACL 2022 robustness、SAM-LML、P-RMF、TMDC、RECAP、EASE。

结论：只做固定比例遮挡或单一高斯噪声已经不够。需要连续强度、多种退化、未知退化和 clean/robust trade-off。

### 簇 C：弱模态优化与样本级贡献

代表：MFON、CVPR 2024 sample-level valuation、EBMC、MRUF。

结论：弱模态优化与逐样本贡献都不是空白。新的方法必须明确区分“采集质量”“任务效用”“跨模态一致性”，并用实验验证这些量不是同一个代理。

### 簇 D：可信度诊断

代表：2026 leakage-safe diagnostic、CPSC。

结论：这是最适合当前项目的切入口，但不能只复现分数置换。更强的贡献应形成完整审计：粒度、单调有效性、混淆不变性、决策作用性和优化不坍缩，并把审计结果用于修复训练机制。

## 4. 对当前项目的直接判断

1. `Q-DAMFON = ALW + DPG + CSS` 作为投稿主方法，创新性和实证强度均不足。
2. 旧结果仍有价值，但应作为“失败原型和机制诊断”，不能宣称稳定优于 MFON。
3. 最可行的重构路线是“诊断优先、干预随后”：先证明常用质量代理在哪些条件下失效，再提出满足审计约束的轻量、可移植修复。
4. 是否达到 CCF-B 水平，取决于能否在至少 MOSI/MOSEI/SIMS、多个骨干、五个随机种子和完整退化协议上形成一致证据，而不是方法名字是否完全脱离 MFON。

## 5. 本项目已经得到的诊断证据

- MOSI 的范数型 CSS 分数与有效时间步数高度相关：train/valid/test 分别为 0.869/0.914/0.896。
- 加性高斯噪声越强，CSS 分数越高；三个 split 的强度-分数相关系数约为 0.969--0.970。
- 最高噪声设置下，99.56%--99.85% 样本的 CSS 分数高于 clean 输入，方向与“高分代表高质量”的假设相反。
- 这些结果由项目根目录 `quality_proxy_audit.py` 复算，尚未涉及模型性能，属于质量代理有效性的机制检查。
