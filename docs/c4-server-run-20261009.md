# C4 单种子归因实验：服务器执行卡

日期：2026-10-09。目标是为第一篇审计论文补齐可靠性损失与逐样本分配的组件归因。此实验只运行 MOSEI seed 1111；结果无论好坏都保留，不根据 test 指标调参。

这里的“Per-sample only”指保留逐样本辅助损失的实现、将分配设为均匀（`constant`）、把可靠性损失权重设为零。它**不会测试非均匀分配是否有效**；将它与 Repaired MFON、Reliability Constant 和 Learned 配对比较，才能逐层拆开实现、可靠性监督和分配的作用。

## 上传文件

把[启动包](../output/server-packages/mfon_c4_attribution_20261009.tar.gz)上传至服务器的 `/home/jovyan/projects/MFON/`。压缩包 SHA-256：`ac8a9497c7591f80bcb7b5e67ed654d9ecf8807e1d2d66f06d093b64a26278c0`；内含训练和测试两个脚本，SHA-256 分别为 `1aad668a751ecfcd69601504c88f585d81d9d1f8ac7d8ee995e3967879eba7d0`、`89900252e768834cb9e73a73434efabc7bffc3ce17b91db8ea012bddd63c4568`。

登录服务器后，**只复制下面代码框中的一整行**，不要复制终端提示符。命令先检查文件哈希；若两个脚本已存在，只在哈希完全相符时复用。脚本自行检查 12 GiB 空间、必要输入、同名输出和 GPU 占用，然后启动后台训练。检查不通过会停止，不会覆盖已有实验。

```bash
bash -c 'set -e; cd ~/projects/MFON; if [ ! -e run_c4_per_sample_only_1111.sh ] && [ ! -e run_c4_per_sample_only_test_1111.sh ]; then printf "%s  %s\n" ac8a9497c7591f80bcb7b5e67ed654d9ecf8807e1d2d66f06d093b64a26278c0 mfon_c4_attribution_20261009.tar.gz | sha256sum -c -; tar -xzf mfon_c4_attribution_20261009.tar.gz -C ~/projects; fi; printf "%s  %s\n" 1aad668a751ecfcd69601504c88f585d81d9d1f8ac7d8ee995e3967879eba7d0 run_c4_per_sample_only_1111.sh 89900252e768834cb9e73a73434efabc7bffc3ce17b91db8ea012bddd63c4568 run_c4_per_sample_only_test_1111.sh | sha256sum -c -; bash run_c4_per_sample_only_1111.sh'
```

查看进度时也只复制命令本身；`tail -F` 会持续显示，按 `Ctrl+C` 只退出查看，不会停止后台训练。

```bash
tail -n 20 -F ~/projects/MFON/mosei_p5_per_sample_only_true_budget_1111.log
```

训练进程结束、日志显示第 25 轮且 checkpoint 存在后，复制下面一行启动 checkpoint 重载测试。测试脚本会拒绝覆盖旧测试日志，并在日志末尾写入 `test_exit_code`。

```bash
bash -c 'set -e; cd ~/projects/MFON; printf "%s  %s\n" 89900252e768834cb9e73a73434efabc7bffc3ce17b91db8ea012bddd63c4568 run_c4_per_sample_only_test_1111.sh | sha256sum -c -; bash run_c4_per_sample_only_test_1111.sh'
```

完成后保存完整训练/测试日志、最终 checkpoint 的 SHA-256、实验命令、退出状态与测试指标。看到 `Epoch:25` 只能说明训练轮次到达末尾，还需要检查进程、checkpoint 和完整测试结果。C4 只有 seed 1111 时，论文仍只能将其称为探索性组件归因。
