# 四源 AnchorCE 相对共享参考的破坏率/恢复率（2026-11-10，本机 CPU）

回应审稿意见：推论 1 的零恢复针对自身封闭头；统一参考（common contract）下的参考是三源核心的封闭头。

- 脚本：`autodl_fscil/anchor_gda_recovery.py`（复制自 anchor_gda.py，仅新增逐样本记录与 D/G 计算，输出到新目录）。
- 本机运行用 `code_cpu/` 中的副本：只把 `cuda` 换成 `cpu`，其余不变。
- 输入：`autodl_frozen_feature_cache_20261109`（特征哈希与归档一致）、`autodl_fscil_dual_20261108/work/readouts`（三源核心参考）。
- 范围：认证轨道 3 数据集 × 3 编码器 × 3 种子 = 27 个任务，261 个会话。
- 核验：261 个会话的无约束准确率、common 契约认证新类/基类准确率与 GPU 归档 `autodl_anchorg_20261109` 逐行一致；每个会话检查 A_C = A_ref − D + G。
- 结果：`summary.txt`（按编码器汇总，9 个划分 bootstrap 20,000 次，种子 2026）；逐划分 `recovery_rows.json`；逐样本 `readouts_anchorg_recovery/**/recovery_sessionXX.npz`。
