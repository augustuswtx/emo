# NCA 大修后完整性审计

日期：2026-09-15  
模式：full（claim / numeric / citation / presentation）

## 审计对象

- `paper/springer-nca/main.tex`、`body.tex`
- `docs/small-paper-draft-v2-en.md`、`docs/small-paper-draft-v2-zh.md`
- `docs/small-paper-references.bib`
- F1--F5 及其 CSV/JSON 源数据
- `output/pdf/nca_anonymous_draft.pdf`
- C1--C5 实验计划、回应稿和匿名工件构建器

## 主张—证据矩阵

| 主张 | 证据 | 判定 |
|---|---|---|
| 早期实现存在批级聚合、提前归约、权重缩小和范数代理反向响应 | 代码审计、既有实验与正文失败分析 | 已支持 |
| 响应头能识别训练内高斯退化 | 三种子 full-test Gaussian AUROC/Spearman | 已支持，但只限训练内扰动 |
| Learned 相对 Constant 的 MAE/Corr/Loss 三种子均值方向有利 | 三个冻结种子的逐种子测试值和配对图 | 已支持为描述性均值，不支持稳定性/显著性 |
| P4 额外训练开销较小且不扩展部署推理图 | 单 GPU、单批次统一微基准与代码路径 | 已支持于该测量协议 |
| 干净绝对分数可跨样本比较并反映辅助目标保真度 | 尚无服务器 C1 结果 | 未支持，正文已标为待检验假设 |
| 正向分配优于 Permuted/Inverse | final-schedule 对照未完成 | 未支持，正文已收缩 |
| 方法稳定、显著或全面优于 MFON | 三种子异质且指标混合 | 不支持，全文已撤除 |
| 一般质量估计、鲁棒融合或模型无关性 | 跨扰动失败、文本主导、单骨干 | 不支持，全文已撤除 |

## 数字一致性

- F2 seed-level CSV 含 3 方法 × 3 seeds × 9 指标的完整值。
- 重新计算的均值和样本标准差与 Table 4、英文稿、中文稿一致。
- Learned−Constant：MAE −0.002467（正文四舍五入 −0.0025）、Corr +0.000700、Loss −0.003993（正文 −0.0040）。
- Gaussian、跨扰动与效率关键数值在摘要、正文、图表和中英文稿中未发现方向冲突。
- MAE/Loss 仅在 Fig. 2d 的“有利方向差值”中反转符号，图注已经披露。

## 引用完整性

- BibTeX 共 31 个唯一条目；LaTeX 正文引用 31 个唯一 key。
- 缺失 key：0；重复 key：0；未引用条目：0。
- 已补充自节奏学习、验证驱动元重加权、多模态课程学习和 DEAR。
- 已删除 7 处 DOI 字段与等价 `doi.org` URL 的重复输出。
- 新增文献的机制定位只用于说明“分配方向不是预算守恒自然推出”，没有借引用替代本研究实验证据。

## 图表与 PDF

- Fig. 1 精简标签并明确无 `q → fusion` 连线。
- Fig. 2 改为逐种子配对点/线和逐种子差值，不再用均值柱高放大小差异。
- 跨扰动表拆为 AUROC 与 ΔMAE 两部分，正常页面尺寸可读。
- Discussion 前设置浮动体屏障，Figs. 3--5 和 Tables 6--7 均在 Discussion 前结束。
- Tectonic 编译通过：25 页 A4；无 LaTeX 错误、缺失文件、未定义引用/交叉引用或 overfull box。
- 25 页已以 contact sheet 全量目检，并重点检查 pp. 5、8、15--18、21；未见裁切、叠字或断裂段落。
- PDF 无自定义元数据、表单或 JavaScript；匿名源码包和代码工件均通过姓名/本机路径标记扫描。

## 剩余严重项

1. C1 冻结检查点的跨样本有效性审计尚未在服务器执行。
2. C2 final-schedule Permuted/Inverse 尚未训练。
3. C4 per-sample-only 组件消融尚未训练。
4. seeds 1114/1115 与 C5 modality-utility 分层尚未完成。
5. Funding、Competing Interests、真实作者贡献、单位、通讯和 ORCID 必须由作者提供，不能代填。
6. 投稿前仍需用常规 TeX Live 或 Editorial Manager 进行最终引擎复核。

## 严重度结论

文字、引用、数字和版式层面的主要问题已经处理；核心科学状态仍为 **Major Revision / 约 5 分**。只有真实完成 C1、C2，并根据结果维持或收缩解释后，才有理由重新评估科学分数。

## 不虚构声明

本轮未生成、补写或推断任何未完成实验结果；所有待运行证据均保持 TBD。
