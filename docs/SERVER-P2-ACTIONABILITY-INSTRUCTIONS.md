# MOSI P2 等预算作用性对照服务器操作

服务器目录：`/home/jovyan/projects/MFON`

## 重要边界

当前25轮 `learned` 训练运行期间不要上传或解压本补丁。等待训练结束、checkpoint保存并完成测试后再更新代码。

## 1. 上传和测试

将 `mfon_actionability_controls_p2_20260730.tar.gz` 上传到 `/home/jovyan/projects`：

```bash
cd /home/jovyan/projects/MFON
tar -xzf ../mfon_actionability_controls_p2_20260730.tar.gz
python -m unittest discover -s tests -v
```

必须显示：

```text
Ran 25 tests
OK
```

## 2. 两轮常数对照烟雾测试

```bash
python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 --epochs 2 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control constant \
  --exp-name p2_constant_smoke \
  2>&1 | tee mosi_p2_constant_smoke_1111.log
```

通过条件：

1. 无 traceback、NaN 或 Inf；
2. `q_v_std`、`q_a_std`为0；
3. 四类权重均值在第2轮为0.1，`w_v_std`和`w_a_std`为0；
4. checkpoint成功保存并可测试。

## 3. 正式对照顺序

只有两轮烟雾测试通过后，按以下优先级运行25轮：

1. `constant`
2. `reversed`
3. `permuted`
4. `oracle`

每个实验仅替换下面两个字段，其他参数与 `learned` 完全相同：

```text
--reliability-allocation-control CONTROL
--exp-name interventional_rel_p2_CONTROL
```

正式命令模板：

```bash
python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control CONTROL \
  --exp-name interventional_rel_p2_CONTROL \
  2>&1 | tee mosi_interventional_rel_p2_CONTROL_1111.log
```

先比较单种子方向，不立即扩展多种子。只有正常 learned 优于常数和错误分数，且 Oracle 给出合理正对照时，才说明可靠性分配具有可审计作用。
