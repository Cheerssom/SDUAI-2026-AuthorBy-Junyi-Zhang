# CIFAR-100 标准 FSCIL 划分实验（Windows 运行说明）

## 目的
回应“缺少标准 FSCIL 协议”的审稿意见：在文献通用的 CIFAR-100 60 基类 + 8×5-way 5-shot 划分（CEC 官方索引）上，用与 CUB 完全相同的 readout、选参网格、基线（SimpleCIL / RanPAC / BiMC / HyCal 迁移）和认证流程评测 Full。

## 0. 准备
- 依赖：torch（CUDA）、transformers、numpy。
- CIFAR-100：`G:\科研新\data\cifar-100-python`（已存在）。
- CEC 官方索引：从 https://github.com/icoz69/CEC-CVPR2021 下载 `data/index_list/cifar100/session_1.txt … session_9.txt`，放到 `G:\科研新\JLT-Head\experiments\expansion_cifar_fscil\index_list\`。脚本会断言 session 2–9 各 25 张、类别依次为 60–64 … 95–99。
- CLIP 权重：transformers 自动下载 `openai/clip-vit-base-patch32` 与 `openai/clip-vit-large-patch14`。

## 1. 特征（每个编码器一次）
在 `G:\科研新\JLT-Head` 下：
```
python src/experiments/expansion_cifar_fscil_features.py --encoder l14 --index_dir experiments/expansion_cifar_fscil/index_list
python src/experiments/expansion_cifar_fscil_features.py --encoder b32 --index_dir experiments/expansion_cifar_fscil/index_list
```

## 2. Readouts（2 编码器 × 2 track × 3 seed = 12 次，可断点续跑）
```
for %e in (l14 b32) do for %t in (standard certified) do for %s in (0 1 2) do python src/experiments/expansion_cifar_fscil_readouts.py --encoder %e --track %t --seed %s
```
（PowerShell：`foreach($e in 'l14','b32'){foreach($t in 'standard','certified'){foreach($s in 0,1,2){python src/experiments/expansion_cifar_fscil_readouts.py --encoder $e --track $t --seed $s}}}`）
- standard：用全部 500 张/类基类训练图拟合（文献协议），报告 9 个会话的 top-1、AIA、PD。
- certified：每个基类预留 10 张 tuning、40 张 calibration 后拟合，报告第 8 次到达后的 common/self 认证新类准确率。
- 显存：RanPAC 随机特征 10000 维，约需 6–8 GB。若 OOM，可把脚本里 `memory_fraction(.7)` 调高。

## 3. 汇总
```
python src/experiments/expansion_cifar_fscil_summary.py
```
输出 `experiments/expansion_cifar_fscil/summary.json`，并打印每个方法的 AIA / last / PD 与 common/self 认证结果。

## 需要回报
1. summary.json 全文；
2. 每个编码器下各方法的 AIA、最后一个会话准确率、PD；
3. certified track 最后一次到达的 common/self 新类准确率与 finite/met 计数。
所有结果无论正负都保留，不改参数、不按测试结果选择。
