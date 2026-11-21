# 独立部署预测器参考：SUN397 重放实验（Windows 运行说明）

## 目的
回应"certification 依赖 Full 自身 closed-head 参考"的质疑：把配对参考换成两个与所有 readout 无关的已部署预测器，其余（offset 候选、δ=.05/5、95% 保持率、cal/test 行）完全不变。

- `common_full`：原共享 Full 参考（对照；脚本会断言与原存档 offset/新类准确率逐条一致）
- `zeroshot`：CLIP zero-shot 基类分类器（E @ T[base]，无训练）
- `probe`：多类 logistic 线性探针，仅用基类 fit 图像；L2 ∈ {1e-2,…,3e-5} 热启动路径，按基类 tune macro 准确率选

## 运行
只需 numpy + scipy（无需 torch/GPU；frozen.pt 由脚本内无 torch 加载器读取）。在 `JLT-Head` 根目录：

```
python src/experiments/expansion_independent_reference_replay.py
```

输入：`experiments/expansion_sun_controls_remote/sun_{l14,b32}.zip`
输出：`experiments/expansion_independent_reference_replay/summary.json` 与 `rows.json`
内存：约 4–6 GB（frozen.pt 解压到内存 + float64 特征）。每个 partition 打印一行进度，共 20 行，然后 `COMPLETE <记录数> <复现数>`；期望 `COMPLETE 3600 1200`。

## 需要回报的内容
1. `summary.json` 中 `reproduced_common` 是否 = 1200（对照复现成功）
2. `references`：每个 partition 下 zeroshot/probe 的 tune macro 准确率
3. `met_all_sessions`：每个参考下满足 95% 目标的次数/总数
4. `summary`（session 5）：各参考 × 方法的 base / novel / closed_base / finite / met
5. `contrasts`：Full − 其他方法 的新类准确率差、95% bootstrap CI、10 个 partition 中 Full 胜出数

所有结果无论正负都保留，不做基于测试标签的选择。
