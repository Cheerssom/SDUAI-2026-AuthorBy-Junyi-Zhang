# VehicleX 车型类别不平衡实验

使用课程数据集列表所链接的 [VehicleX 官方仓库](https://github.com/yorkeyao/VehicleX) 中的 VeRi 适配版。下载文件名为 VeRi_ReID_Simulation.zip。原始图片和标签放在 ZIP 内，脚本从 VeRi_label.xml 读取车型和车辆 ID。ZIP 的 SHA-256 为 213bf03bffad83f5d1084d329fa8da26743378a5d51aadde7f8c83f077cdf52c。

全量 75,516 张图片，1,362 个车辆 ID，11 个车型类别。图像频次最大为 16,036，最小为 836。用种子 2026 每个车辆 ID 和车型组合最多抽取 24 个视角，本版共选 32,688 张。每个种子取五折 StratifiedGroupKFold 的第一折作测试，车辆 ID 在训练、测试间不重复。对比普通、逆频率加权及类内有放回过采样的逻辑回归，三种方法使用同一份 CLIP 特征与同一划分，固定 C=1。尾类按该折训练频次定义为最少的三类。报告准确率、macro-F1、macro recall、尾类平均 recall 和各类 recall。三个种子共享数据集，统计量用来展示划分敏感性。

在 PyTorch 2.8.0 / CUDA 12.8 环境安装 torchvision、Pillow、NumPy 和 scikit-learn 1.7.2，并准备 [OpenAI CLIP 的 model.py](https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py)（保存为 openai_clip_model.py）以及 [官方 ViT-B/16 权重](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)。先检查标注，再运行。

~~~bash
python run.py VeRi_ReID_Simulation.zip --inspect
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u run.py VeRi_ReID_Simulation.zip \
  --attribute type --weights ViT-B-16.pt --cap 24 --batch-size 64 --output results
~~~

结果生成在 results/metrics.json，包含原始类别频次、各划分的训练和测试类频次、各方法的指标。results/samples.json 记录入选图像，results/features.npy 是逐行对齐的特征缓存。

## AnchorCE 类别增量实验

run_anchor_vehiclex.py 读取上一步保存的 samples.json 和 features.npy，复用 JLT-Head/autodl_fscil 下的 anchor_gda.py、fscil_readouts.py、extra_baselines.py、kbs_paired_accuracy.py 和 kbs_risk_control.py。使用同一 OpenAI CLIP ViT-B/16 权重与 [CLIP tokenizer 文件](https://huggingface.co/openai/clip-vit-base-patch16/tree/57c216476eefef5ab752ec549e440a49ae4ae5f3)，车型编号与官方 list_type.txt 对应。

按车辆数选定六个基类（sedan、SUV、hatchback、bus、truck、sports car），另外五类在每个种子下随机排序，每个阶段增加一类。每一新类用五辆不同车辆各提供一张支持图。每类约 20% 的车辆用于测试；对于基类，先独立保留八辆用于调参、四十辆用于校准，其余车辆用于拟合。每辆校准车辆只取一张图。三组种子为 42、43、44。所有数据角色均互斥于车辆 ID。

AnchorCE 使用 ridge、文本、视觉原型和基类冻结协方差四个分支。融合参数通过基类内部的伪增量任务选择。由于基类库小，部署阶段按论文的小参考库规则使用 η=6/30，伪增量调参用 η=3/30。GDA 每次扩类后重新估计协方差，GDA-Anchor 在基类上冻结协方差并使用同样的小参考库规则。三者共享图像特征、文本锚点、支持图和校准集。

~~~bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u run_anchor_vehiclex.py \
  --source /path/to/JLT-Head/autodl_fscil --cache results --weights ViT-B-16.pt \
  --tokenizer clip_tokenizer --output anchorce_results
~~~

结果记录末阶段及中间各阶段的基类、新类和宏平均准确率。共同参考采用 AnchorCE 第零阶段的闭集头，自身参考采用各方法自己的闭集头。每类校准预算为 5、10 或 40 张，目标为保留参考准确率的 95%，总显著性水平 0.05 分配到五个增量阶段。回退时新类准确率计零。每次运行保存支持图索引、调参选择和各数据角色的车辆 ID。

在上述命令中加入 --require-prototype 并将 --output 改为 anchorce_prototype_results，可运行视觉原型对照。固定原先选出的 ridge 与协方差权重，在同一基类伪增量任务上，从 {0.25, 0.5, 1, 2, 4} 中选择原型权重，并重新选择准入偏置。数据划分、支持图和认证设置沿用原实验。

加入 --require-prototype --compare-admission --seeds 42 43 44 45 46 47，并将 --output 改为 anchorce_dense_results，可分别评估三种新类准入改动。基类间隔版从每个新类分数中减去基类最高分与第二高分之差乘以系数，系数在基类内部的伪增量任务上选取，以保留 95% 基类准确率时的新类准确率为目标。固定下注版沿用原始分类分数及候选偏置，从原下注网格预先固定 λ=1。候选加密版加入 1,001 个调参集分位数和低分位端的八个更保守候选。三种变体各自校准，使用相同的保留比率、显著性阈值和固定顺序检验流程。所有候选偏置均由调参集分差生成，在校准之前固定。六个种子的结果可按任意子集汇总。

使用 --require-prototype --dense-offsets-all --seeds 42 43 44 45 46 47 对三种评分方法统一应用候选加密与保守端扩展规则。各方法的候选偏置从各自的基类调参分差产生，共用数据划分、校准预算和认证检验。共同参考下三种方法均保护 AnchorCE 的初始闭集头，自己的参考下保护各自的闭集头。结果见 anchorce_matched_metrics.json。

使用 --require-prototype --compare-admission --fscil-baselines --seeds 42 43 44 45 46 47，可将对照换为 HyCal 和 TEEN。它们采用项目中同一冻结 CLIP 特征上的原始打分迁移实现及 101 分位候选。以 AnchorCE-DenseOffsets 结果行与这两条对照比较。结果见 anchorce_fscil_baselines_metrics.json。

## 论文与补充材料

论文源码在 paper_cvpr/。main.tex 为正文（CVPR 模板，正文 8 页），supplementary.tex 为补充材料，VehicleX 实验见正文表 6 与补充材料 G 节。VehicleX 相关的表格和图由脚本直接从上述 JSON 结果生成，不需要 GPU：

~~~bash
MPLCONFIGDIR=/tmp/mpl python paper/make_tables_figures.py
~~~

脚本读取 anchorce_fscil_baselines_metrics.json、anchorce_dense_metrics.json、anchorce_matched_metrics.json 和 metrics.json，写出 paper_cvpr/table_vehiclex_deployed.tex、table_vehiclex_dataset.tex、table_vehiclex_perseed.tex、table_vehiclex_budget.tex 与 fig_vehiclex_sessions.pdf，并在 paper/numbers.json 中记录文中引用的全部统计量（含校准预算下界 k=129 的数值求解）。之后在 paper_cvpr/ 下依次编译：

~~~bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
~~~

补充材料通过 xr-hyper 引用正文编号，需先编译 main。
