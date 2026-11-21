# AnchorCE：代码与实验结果

本仓库对应论文《AnchorCE: Base-Anchored Evidence Alignment for Retention-Certified Few-Shot Class-Incremental Learning》（作者 Junyi Zhang）的正文与补充材料，包含全部实验代码、逐次运行结果和汇总结果（论文正文不在本仓库中）。补充材料 PDF 位于根目录 `supplementary.pdf`。

## 目录结构

```
supplementary.pdf           论文补充材料
course_vehiclex/            VehicleX 车型增量实验（正文表 6、补充材料 G 节）
  README.md                 环境、数据准备与运行命令
  run.py                    VehicleX 数据检查与 CLIP ViT-B/16 特征提取
  run_anchor_vehiclex.py    AnchorCE / GDA / GDA-Anchor / HyCal / TEEN 增量与认证实验
  paper/make_tables_figures.py   由下列 JSON 生成正文表 6、表 S23–S25 与图 S3
  *_metrics.json, *.log     六个种子（42–47）的完整结果与运行日志
  samples.json              入选图像清单（与特征逐行对齐）
cvpr_fscil/
  code/autodl_fscil/        标准 FSCIL 基准主流程（特征提取、各方法读出、配对认证）
  code/src/                 留出域、外部分类器、恢复耦合等补充实验代码
  code/paper_scripts/       论文表格与图的生成脚本（build_*.py、figures_src/）
  results/experiments/      每项实验的协议文档、逐次运行结果（JSON/NPZ）与汇总（summary.json、verification.json）
```

## 论文内容与结果目录

| 论文内容 | 结果目录（`cvpr_fscil/results/experiments/` 下） |
|---|---|
| 正文表 1–3、图 4，补充表 S1–S4、S6–S7、S9–S13（CEC 三基准、三种编码器、common/self 两种参考、逐会话、LLM 描述对照） | `merged_ref4_20261110`、`autodl_round4_20261109`、`autodl_round5_20261110`、`autodl_anchorg_20261109`、`autodl_extra_baselines_20261109`、`autodl_descriptions_20261109`、`expansion_shared_head_remote`、`expansion_self_reference_remote`、`expansion_hycal_self_remote`、`expansion_cub_remote`、`expansion_reference_completion` |
| 正文表 4（来源与对齐消融） | 同上（`merged_ref4_20261110`、`expansion_shared_head_remote`） |
| 正文表 5，补充表 S5、S8（校准预算、外部 logistic probe、B/32 与 L/14 probe、MLP、200 次校准重采样） | `external_probe_encoder_validation_20261117`、`external_mlp_validation_20261117`、`expansion_equal_budget_external_smoke`、`expansion_prospective_external_smoke`、`expansion_followup_diagnostics` |
| 正文图 3c、3d，补充表 S18（固定准确率下的恢复耦合、方差斜率、固定边际交换） | `kbs_mechanism_followup`、`expansion_followup_diagnostics`、`local_recovery_20261110` |
| 补充 C 节，表 S14–S16（独立等网格重调、小参考库收缩、插值常数） | `kbs_full_fusion_retuning`、`expansion_shrinkage`、`expansion_constant_sensitivity_remote`、`expansion_reference_cardinality`、`expansion_cardinality_rule_v2` |
| 补充 D 节，表 S17（闭式边界与学习式路由器） | `expansion_gate_calibration` |
| 补充 E 节，表 S19–S20（ImageNet-Sketch 迁移、交叉熵头、SigLIP、ResNet-50） | `kbs_sketch_acquisition`、`kbs_sketch_validation`、`kbs_imagenetr`、`expansion_cross_encoder`、`expansion_rn50`、`expansion_softmax_readout` |
| 补充 F 节，表 S21–S22、图 S2（SUN397、Food-101、查询矩置换、四种受保护参考） | `expansion_prospective_validation`、`expansion_sun_controls_remote`、`expansion_independent_reference_replay`、`kbs_independent_risk`、`kbs_reviewer_revision` |
| 其他补充分析 | `kbs_strengthening`、`kbs_retention_safety_diagnostic`、`fcs`、`fcs_pets` |

每项实验的设定写在 `results/experiments/` 顶层的 `*_protocol.md` 中；多数目录含 `verification.json`，记录特征与预测的独立重建校验。

## 环境

Python 3.10+，PyTorch 2.x（CUDA），transformers、numpy、scipy、scikit-learn、matplotlib、Pillow。编码器为 OpenAI CLIP ViT-B/16、ViT-L/14、ViT-B/32（另有 RN50、SigLIP 对照），均冻结使用。

## 数据

- CIFAR-100、miniImageNet、CUB-200：按 CEC 官方划分（https://github.com/icoz69/CEC-CVPR2021 的 `data/index_list/`），放置方式见 `cvpr_fscil/code/autodl_fscil/README_ZH.md`。
- SUN397、Food-101、ImageNet-R、ImageNet-Sketch、Oxford-IIIT Pets：官方发布版本。
- VehicleX：课程数据集列表中的官方仓库 https://github.com/yorkeyao/VehicleX，VeRi 适配版 `VeRi_ReID_Simulation.zip`，见 `course_vehiclex/README.md`。

## 运行

标准 FSCIL 基准（特征提取与全部读出，可断点续跑）：

```bash
cd cvpr_fscil/code/autodl_fscil
bash run_all.sh
```

VehicleX 实验（需先按 `course_vehiclex/README.md` 提取特征）：

```bash
cd course_vehiclex
python run_anchor_vehiclex.py --source ../cvpr_fscil/code/autodl_fscil --cache results \
  --weights ViT-B-16.pt --tokenizer clip_tokenizer --require-prototype --compare-admission \
  --fscil-baselines --seeds 42 43 44 45 46 47 --output anchorce_fscil_baselines_results
python paper/make_tables_figures.py
```

`make_tables_figures.py` 不需要 GPU，直接由本包中的 JSON 重新生成 VehicleX 的全部表格与图。

## 随机种子

VehicleX：样本抽取种子 2026，实验种子 42–47。CEC 基准：官方划分与三个划分种子；补充实验的划分、顺序与重采样种子记录在各自的协议文档与结果 JSON 中。

## 未包含的内容

CLIP 特征缓存（`*.pt`，约 21 GB）、数据集与模型压缩包（约 6 GB）以及表格脚本不读取的中间分数缓存（`*.npz`）未放入本仓库；它们可由上述代码与公开数据重新生成。用于远程集群调度的脚本（SSH 连接、文件同步）不属于实验代码，也未放入。
