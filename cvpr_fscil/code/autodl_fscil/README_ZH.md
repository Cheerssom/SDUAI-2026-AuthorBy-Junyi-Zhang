# 标准 FSCIL 基准实验（AutoDL RTX 4090）

## 目的
按领域惯例补齐实验（见 research_notes/20261108_teacher_audio/related_work_survey_zh.md）：
- **E1** 标准三件套 CIFAR-100、miniImageNet、CUB-200（CEC 官方划分），CLIP **ViT-B/16**（另跑 L/14、B/32）。
- **E2** 新增免训练基线 CLIP Zero-Shot、TEEN、FeCAM，与 AnchorCE、None、Fixed、Dense、SimpleCIL、RanPAC、BiMC、HyCal 同特征、同支持集。
- **E3** 领域通用指标：逐会话 top-1、Avg、Last、PD、三数据集 Average；本文新增 common/self 认证新类准确率。

## 1. 环境
```bash
source /etc/network_turbo          # AutoDL 学术加速（GitHub / HuggingFace）
export HF_ENDPOINT=https://hf-mirror.com   # 若 HuggingFace 仍慢
pip install -U transformers pillow
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

## 2. 代码
把整个 `autodl_fscil/` 上传到 `/root/autodl-tmp/autodl_fscil/`。

## 3. 数据（放在 /root/autodl-tmp/data/）
```
data/
  cifar-100-python/            # 官方 python 版：train, test, meta
  miniimagenet/images/*.jpg    # CEC 版 miniImageNet（100 类，每类 600 张，84×84）
  miniimagenet/split/train.csv, test.csv   # 列：filename,label(wnid)
  miniimagenet/class_names.txt # 可选：每行 "wnid 类名"，提供后文本锚点用可读类名
  cub/CUB_200_2011/...         # 官方 CUB_200_2011（images.txt、classes.txt 等）
  index_list/cifar100/session_1..9.txt
  index_list/mini_imagenet/session_1..9.txt
  index_list/cub200/session_1..11.txt
```
- **CIFAR-100**：https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- **CUB-200-2011**：Caltech 官方 CUB_200_2011.tgz（约 1.1 GB）。
- **miniImageNet 与三套 index_list**：CEC 官方仓库 https://github.com/icoz69/CEC-CVPR2021 的 `data/index_list/` 及其 README 给出的 miniImageNet 数据。
- **miniImageNet 类名**：若无 class_names.txt，文本锚点退化为 wnid，CLIP Zero-Shot/文本分支会显著变差。请用 ImageNet 的 wnid→类名表（如 `LOC_synset_mapping.txt`，格式 "n01532829 house finch, linnet, ..."）生成该文件；脚本取逗号前第一个名字。
- 脚本会断言每个增量会话的类别与 5 张/类数量与 CEC 划分一致，不一致会直接报错。

## 4. 运行（可断点续跑）
```bash
cd /root/autodl-tmp/autodl_fscil
nohup bash run_all.sh > work/run.log 2>&1 &
tail -f work/run.log
```
- 只跑 ViT-B/16：`ENCODERS="b16" bash run_all.sh`
- 暂停：`touch work/PAUSE`；继续：删除该文件后重新运行 run_all.sh（已完成部分自动跳过）。
- 预计：特征 ~1 小时（含 L/14），readout 54 组 2–4 小时。

## 5. 回报
完成后把 `work/summary.json` 与 `work/summary.txt` 下载回 `G:\科研新\JLT-Head\experiments\fscil_standard\`，并回报：
1. 每个数据集 × 编码器的各方法 Avg / Last / PD 与逐会话准确率；
2. 三数据集 Average；
3. certified track 最终会话 common/self 认证新类准确率与 finite/met；
4. 任何报错或断言失败的完整信息。
所有结果无论正负都保留，不改参数、不按测试结果选择。

## 实现说明（论文中需注明）
- 所有基线为冻结 CLIP 特征上的**匹配迁移实现**，与 AnchorCE 共享特征、支持集与调参访问。
- TEEN：新类原型 = 0.9·原型 + 0.1·Σ softmax(16·cos(原型, 基类原型))·基类原型。
- FeCAM：每类收缩协方差（对角/非对角各加均值收缩，γ1=γ2=1）、相关化归一、Mahalanobis 距离取负为分数。
- BiMC 的描述分支在无类别描述时退化为类名文本。
- 认证轨道：CIFAR/mini 每基类预留 10 tuning + 40 calibration；CUB 预留 5 + 10（与已有 CUB 实验一致）。

## 6. FeCAM 修正后的重跑（2026-11-08）
首轮 FeCAM 让每个新类用 5 张支持图单独估计 512/768 维协方差，距离尺度失真，随会话持续下滑（miniImageNet B/16 第 0 会话 91.5 → 末会话 58.2），与 BiMC 论文报告的 FeCAM 水平明显不符。修正：基类用各自协方差，少样本新类用**基类协方差均值**（各自均值），其余收缩与相关化步骤不变。
只需重跑 FeCAM，其他 10 种方法的记录原样保留：
```bash
cd /root/autodl-tmp/autodl_fscil
cp -r work/readouts work/readouts_backup_before_fecam   # 先备份
nohup bash rerun_fecam.sh > work/fecam_rerun.log 2>&1 &
```
（若结果分布在 node1/node2 两台机器，分别在各自的 work 上运行，再按原方式合并汇总。）
完成后回传新的 summary.json / summary.txt，并报告 FeCAM 在三个数据集 B/16 下的逐会话准确率。

### 本次执行核验补充

`--only FeCAM_transfer` 已在入口直接进入 `fecam_replay.py`，仅使用已归档的 fit/tune/cal 划分和 Full/FeCAM 闭集正确性，不再拟合 Ridge、重新调融合参数或运行 RanPAC。此模式强制 CPU；每任务 `fecam_patch.json` 记录 CUDA 未初始化、版本、特征散列和不变文件散列。

启动需一并上传 `fecam_replay.py`、`fecam_replay_queue.py`；shell 使用独立 venv Python 和 `set -euo pipefail`。队列仅处理已存在完整结果的任务，不创建缺失实验。

本次将第二节点的24组原始记录导入第一节点，与其30组合并后，在第一节点以四任务、每任务四 CPU 线程完成54组专项更新。原结果保留在 `experiments/autodl_fscil_dual_20261108`；更新结果自动回传到 `experiments/autodl_fecam_replay_20261108`，同时核验其余10方法逐记录完全相同，selection.json、complete.json、closed_references.npz 字节不变。

基类协方差均值规则作为新的 FeCAM-style 协方差迁移版本记录；初次独立新类协方差结果仍完整保留，不凭预期结果认定原论文复现。

## 7. FeCAM 第二次修正（2026-11-08）
第一次修正后 miniImageNet / CIFAR 明显改善，但 CUB 第 1 会话从 73.8 跌到 37.8：CUB 基类仅约 30 张/类，自身协方差与新类使用的基类平均协方差尺度不一致，新类距离系统性偏小。
修正：所有类别统一使用正则化判别分析估计 Σ_c = (n_c·S_c + m·S_pool)/(n_c + m)，m 为特征维度，S_pool 为基类协方差均值；其后收缩、相关化、Mahalanobis 不变。
重跑方式与第 6 节相同（先备份当前 work/readouts，再 `nohup bash rerun_fecam.sh > work/fecam_rerun2.log 2>&1 &`），只更新 FeCAM。

## 8. 类别描述实验（2026-11-09）
目的：BiMC 原文使用 LLM 生成的类别描述，其报告数字高于我们的“仅类名”迁移版本。本实验让**所有使用文本的方法获得同一套描述**（AnchorCE、Zero-shot、Dense、No-centering、Fixed、HyCal 的文本锚点 = 0.5·类名模板 + 0.5·描述；BiMC 按其原式使用描述），在相同条件下比较。现有结果不受影响，新结果写入 `work/readouts_desc/` 与 `work/summary_desc.json`。

### 8.1 准备描述文件 `descriptions/<数据集>.json`（cifar100 / mini / cub200）
格式：`{"类名": ["描述句1", "描述句2", ...], ...}`，类名须与 `work/features/<ds>/<enc>/frozen.pt` 中 `names` 一致（大小写、下划线不敏感；miniImageNet 若 names 是 wnid，则键也用 wnid）。
1. **优先**：查找 BiMC（CVPR 2025, Chen et al., "Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration"）官方代码仓库中发布的三个数据集描述文件，转换为上述格式；记录来源 URL 与 commit。
2. **若找不到**：用同一个 LLM、同一提示词为每类生成 10 句描述，温度 0，记录模型名与日期。提示词：
   `Describe what a {class name} looks like in a photo. Give 10 short, distinct visual descriptions, one per line, each under 20 words, without numbering.`
   CUB 类名用可读鸟名；miniImageNet 用 wnid 对应的 ImageNet 类名。
3. 每类至少 1 句；不得使用测试图片或测试结果来挑选描述。

### 8.2 运行
```bash
cd /root/autodl-tmp/autodl_fscil
nohup bash run_desc.sh > work/desc_run.log 2>&1 &
```
只跑 B/16：`ENCODERS="b16" bash run_desc.sh`。时间约与首轮 54 组相当（可按原双节点方式拆分）。

### 8.3 回报
- `work/summary_desc.json`、`work/summary_desc.txt`；
- 描述来源（BiMC 官方 / 自行生成及模型、提示词）；
- 与 `summary.json`（仅类名）对照：各方法 B/16 三数据集 Avg / Last / 认证准确率的变化。
所有结果无论正负都保留。

## 9. 第三轮实验（2026-11-09）：A 新基线 GDA+KLDA，B r/校准预算扫描，C 效率

一条命令跑完三部分（可断点续跑，已完成的任务自动跳过）：
```bash
tmux new -s round3
bash run_extra.sh                 # 默认 STAGES="A B C"；也可只跑某部分，如 STAGES="B" bash run_extra.sh
```
不会改动已有结果：A 只读 `work/readouts/`，写 `work/readouts_extra/`；B 用 `--tag _sweep` 写到 `work/readouts_sweep/`；C 只计时，写 `work/efficiency.json`。

- **A. GDA（ICLR24）+ KLDA（AAAI25）**：闭式免训练基线，超参只在基类调参集上选；先跑合成数据自检，必须打印 `SELFTEST gda_acc=… klda_acc=…` 且不报错。54 个任务，约 1–2 小时。
- **B. 扫描**：B/16、certified track、3 数据集 × 3 seed，重新跑完整读出，并在每个会话对 r∈{.90,.95,.99} × 每类校准图像数∈{5,10,20,40}（CUB 最多 10）再认证一次。r=.95 且校准数取满时应与原主结果一致（用于核对）。约 2–3 小时（FeCAM 在 CPU 上较慢）。
- **C. 效率**：miniImageNet 与 CUB-200、B/16，每个方法每个增量会话"吸收新类 + 对全部测试集打分"的耗时（预热 1 次 + 重复 3 次取中位数）、基类拟合耗时、峰值显存、需存储的状态大小。约 20–30 分钟。**跑 C 时 GPU 上不要有其他任务**，否则计时不准。

### 9.1 出错处理
任何 assert / 报错 / OOM：立即停止，把完整报错发回，不要自行修改代码或参数。

### 9.2 回报（发回这些）
1. `tail -40 work/round3.log`；统计 `EXTRA_COMPLETE`+`EXTRA_ALREADY_COMPLETE` 数（应 54）。
2. `grep -E "GDA|KLDA|/full" work/summary_with_extra.txt`
3. `cat work/summary_sweep.txt`
4. `cat work/efficiency.txt`
5. 打包传回本地 `experiments/autodl_round3_20261109/`：
   `tar czf round3_results.tgz work/readouts_extra work/readouts_sweep work/summary.json work/summary_with_extra.txt work/summary_sweep.json work/summary_sweep.txt work/efficiency.json work/efficiency.txt work/round3.log`

## 10. AnchorCE + 基类冻结 LDA 证据源（2026-11-09，优先于第 9 节 B/C）
`anchor_gda.py`：在 AnchorCE 的文本/岭回归/原型三源之外，加入只用基类数据估计后冻结的共享协方差 LDA 分数，同样按基类库逐查询标准化，系数 g∈{0,.25,.5,1,2} 与 (a,u,d) 一起在基类伪扩展上选（g=0 即原 AnchorCE）。
内置两道核对：① g=0 时的选择必须与归档的 AnchorCE 选择一致；② 每个会话检查基类分数不变（零恢复）。任一 assert 失败立即停止并回报。
只读 `work/readouts/`，结果写 `work/readouts_anchorg/`，方法名 `full_gda`。54 个任务，约 1–2 小时。
```bash
tmux new -s anchorg
bash run_anchorg.sh
```
回报：`tail -40 work/anchorg.log`（每个任务一行 ANCHORG_COMPLETE，含选出的 g）；`grep -E "GDA_transfer|/full|full_gda" work/summary_with_anchorg.txt`；打包
`tar czf anchorg_results.tgz work/readouts_anchorg work/summary.json work/summary_with_anchorg.txt work/anchorg.log` 传回本地 `experiments/autodl_anchorg_20261109/`。

## 11. 第四轮（2026-11-09）：四源 AnchorCE 的消融 / 描述 / 扫描 / 效率
一条命令，分 A–D 四段，可断点续跑（已完成任务自动跳过）；可用 `STAGES="A B" bash run_round4.sh` 只跑部分。
```bash
tmux new -s round4
bash run_round4.sh
```
- **A 消融**（3 编码器 × 3 数据集 × 2 track × 3 seed × 3 变体 = 162 个任务）：`none`（四源都不中心化）、`fixed`（四源都用固定矩）、`rawg`（三源锚定，但 LDA 分数不锚定直接加）。每个任务都核对 g=0 时与归档的同类三源读出选择一致、基类分数不变。
- **B 描述**（54 个任务）：四源 AnchorCE 用 BiMC 官方描述（读 `work/readouts_desc/` 与 `work/description_features/`，需已存在）。
- **C 扫描**（B/16 certified，3 数据集 × 3 seed）：11 个原读出（`fscil_readouts.py --sweep`，FeCAM 在 CPU 上较慢）、GDA/KLDA、四源 AnchorCE，各在 r∈{.90,.95,.99} × 每类校准数∈{5,10,20,40}（CUB 最多 10）上认证。
- **D 效率**（miniImageNet、CUB-200，B/16）：**跑 D 时 GPU 上不要有其他任务**。
预计总时长 5–8 小时（A 最多）。只写新目录：`work/readouts_anchorg_{none,fixed,rawg}`、`work/readouts_anchorg_desc`、`work/readouts_sweep`、`work/readouts_extra_sweep`、`work/readouts_anchorg_sweep`、`work/efficiency.json`。

回报：
1. `tail -60 work/round4.log`；统计 `ANCHORG_COMPLETE` 行数（A 应 162，B 应 54，C 应 9）与 `EXTRA_COMPLETE`（C 应 9）。
2. `grep -E "/full |/full_gda|GDA_transfer" work/summary_round4.txt`（含 AVERAGE 行），以及 `grep AVERAGE work/summary_round4.txt`
3. 描述汇总：`python fscil_summary.py _desc > work/summary_desc_round4.txt; grep -E "AVERAGE|full_gda|BiMC" work/summary_desc_round4.txt`
4. `cat work/summary_sweep.txt`、`cat work/efficiency.txt`
5. 打包传回 `experiments/autodl_round4_20261109/`：
   `tar czf round4_results.tgz work/readouts_anchorg_none work/readouts_anchorg_fixed work/readouts_anchorg_rawg work/readouts_anchorg_desc work/readouts_sweep work/readouts_extra_sweep work/readouts_anchorg_sweep work/summary*.json work/summary*.txt work/efficiency.json work/efficiency.txt work/round4.log`

## 12. 第五轮（2026-11-10）：即插即用锚定 + 以四源 AnchorCE 闭集头为部署参照
```bash
tmux new -s round5
bash run_round5.sh                  # 默认 STAGES="A B C D"，可断点续跑
```
- **A 即插即用**（B/16，3 数据集 × 2 track × 3 seed = 18 任务，约 1 小时）：HyCal、GDA 的原版与"基类锚定版"（HyCal 的逐查询矩只用基类；GDA 的协方差只用基类并冻结、分数相对基类标准化），记录每个会话相对自身会话 0 的恢复/破坏率，认证同时报告共同（四源闭集头）与自身契约及校准预算扫描。
- **B/C/D 换参照重认证**（只跑 certified track）：共同契约的参照由三源核心闭集头换成四源 AnchorCE 闭集头（读取 `work/readouts_anchorg/.../closed_references.npz` 中的 `full_gda_*`）。B=四源 AnchorCE 自身（27 任务）、C=GDA/KLDA（27）、D=11 个原读出（27，FeCAM 在 CPU 上最慢，约 3–4 小时）。每个认证任务同时保存 r×校准数扫描。
- 只写新目录：`work/readouts_plugin`、`work/readouts_anchorg_ref4`、`work/readouts_extra_ref4`、`work/readouts_ref4`；不改动任何已有目录。
- 任何 assert / 报错 / OOM：立即停止并回报，不要自行改代码。

回报：
1. `tail -60 work/round5.log`；统计 `PLUGIN_COMPLETE`（应 18）、`ANCHORG_COMPLETE`（B 应 27）、`EXTRA_COMPLETE`（C 应 27）、`FSCIL_READOUT ... certified` 的完成任务数（D 应 27 个 complete.json）。
2. `cat work/summary_plugin.txt`
3. `grep -E "CERTIFIED|AVERAGE" work/summary_ref4.txt`
4. `cat work/summary_sweep_ref4.txt | grep -E "b16/(full_gda|GDA_transfer|GDA_anchor|HyCal_transfer|HyCal_anchor|FeCAM_transfer|BiMC_transfer|dense)/0.95_(5|10)"`
5. 打包传回 `experiments/autodl_round5_20261110/`：
   `tar czf round5_results.tgz work/readouts_plugin work/readouts_anchorg_ref4 work/readouts_extra_ref4 work/readouts_ref4 work/summary_plugin.* work/summary_ref4.* work/summary_sweep_ref4.* work/round5.log`
