# P3：公开 filament-seg 方法与验证协议

读取日期：2026-10-30（Asia/Singapore）。仓库：
[ameyypawar/filament-seg](https://github.com/ameyypawar/filament-seg)。
固定提交：`81a739bb7786a44b989f5d76afbb6f9d22b5d1dd`（提交日期 2026-10-08）。
性质：Codex 方法调查；涉及方法优劣的判断为 **same-family / provisional**，不是独立验证。
只下载和读取公开源码、许可证及报告 TeX，没有执行第三方代码、下载其训练权重、MAGFiLO 标注或 submission。

## 证据保存

- `commit.json`、`tree.json`：当时仓库提交和文件树。
- `LICENSE`：MIT，Copyright (c) 2026 Amey Pawar。
- `source_manifest.json`：每个所读文件的 URL 和 SHA256。
- `source/`：按上述提交固定的关键训练、预处理、模型、融合和评估源码。
- `ultralytics_LICENSE`、`smp_LICENSE`、`torchvision_LICENSE`：其依赖的官方上游许可证快照。

下文行号均指本目录保存的对应源文件；链接指固定提交，后续仓库变化不会改变本次依据。

## 方法事实

| 组成 | 源码实际行为 | 可借鉴之处 |
|---|---|---|
| 原始输入 | 2048² 灰度；网络训练截取 **512² 原分辨率 crop**，不是把整幅图 resize 到512² | 保留几像素宽细丝；应先于单纯增大 epoch 解决尺度损失 |
| 正负采样 | 每个训练观察每 epoch 默认8 crop；70%以某个 GT 实例的 bbox 中心为目标，并在 ±crop/4 范围抖动；其余按太阳盘面积均匀采样 | 前景裁块与背景裁块都有明确角色，避免训练被大背景主导 |
| 目标 | 默认每张观察选一个 annotator 的二值 union；另外实现逐图 soft consensus 和 all-annotator union | 不把“支持多种 target”误当成最终模型使用 consensus；其报告最终配方是 single-annotator |
| 几何增强 | flips、90° rotation，亮度/对比度只作用 intensity；radius 和 mask 同步几何变换 | 不使用缩放、elastic 等会消掉细丝的重采样增强 |
| 输入归一化 | 拟合太阳盘，按半径的128个环带求强度中位数，平滑径向 profile，图像除以该 profile；clip ratio 到[0,2]后存uint8；加载 intensity 为该 ratio/2 | 径向光度变化与局部暗丝分离；应记录归一化定义，而不是笼统写“除255” |
| 第二通道 | 距太阳盘中心的距离除以拟合半径；相同 crop/增强作用在两通道上 | 将径向位置作为图像几何坐标；不是台站或观测时间 metadata |
| 网络/损失 | `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1)`；BCEWithLogits + soft Dice；AdamW lr1e-3，cosine | 架构、预训练、采样和归一化是多因素变化，不能把差异全部归功于 ResNet34 |
| 最终训练配方 | 报告写30 epoch、batch16、每观察8 crop、单 annotator、最后一轮权重 | 是来源中的已用配方；不是我方现在应盲目复制的训练预算 |
| 语义推理 | 512² tile，overlap128，二维 Hann 权重混合 **logits**；8种D4翻转/旋转的 logits 再平均 | 与全图缩小推理不同，成本明显更高；初轮可固定无TTA以隔离分辨率作用 |

证据：
[dataset.py L139–175、188–211、276–300](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/dataset.py#L139-L300)，
[preprocess.py L54–58、80–122](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/preprocess.py#L54-L122)，
[disk.py L249–290](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/disk.py#L249-L290)，
[model.py L23–40、104–151、191–212](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/model.py#L23-L212)，
[train.py L136–179](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/train.py#L136-L179)，
[报告 L99–109](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/reports/main.tex#L99-L109)。

### 太阳盘拟合的具体内容与实现边界

作者不单独依靠亮度阈值。`disk.py` L77–116 从强度阈值、局部纹理和标准 GONG 几何给出起点，向内移动若干距离，在720条径向射线上找最强边缘并拟合圆，选择具有足够射线支持的最小圆，避免把明亮 halo 外缘当成 limb。盘掩码默认缩到0.98R。

这是可研究的物理处理，但源码同时含标准几何兜底、无候选返回、空 profile 补偿等分支（例如 L110–114、167–168、235–238、267–274）。我方应借鉴清晰的数据变换，不整段复制这些与用户“禁止防御性编程”冲突的回退逻辑；真实失败应记录并修根因。其 halo 半径差是作者在其全体图像上计算的统计，未在本项目复测，也不当成本项目的测量。

## YOLO11-seg 与实例融合

检测器以 limb-corrected 灰度帧作为单类图像，每个 annotator 的 drawing 各成训练样本（同一个原始像素帧可出现多个视图）；polygon 坐标归一化为[0,1]。
报告的三个模型为 YOLO11s/1024/60 epoch、YOLO11m/1280/80 epoch、YOLO11m/1536/80 epoch。Mosaic、mixup 和 copy-paste 关闭，保留翻转、亮度和少量平移/尺度增强。检测 mask 本身较粗，不能把其上采样后的尺寸误当成原生细边界精度。

融合明确分两步：

1. 不同模型或视图的 mask IoU >0.5 时，按最高置信度候选进行合并，保留该 mask；置信度按所有模型/视图平均，漏检的一方计0。这个0.5是**检测器合并参数**，不是允许重复实例的官方匹配漏洞。
2. U-Net probability 阈值化、闭运算、填洞并限于太阳盘；高置信度检测先领取其扩张 mask 中尚未被领取的 U-Net 前景。一个 detection 可领取多个不连通的语义组件而形成一个实例；实例互斥，最终边界来自语义前景。报告最终设置为 threshold0.5、close radius3、detection score≥0.30、grow4、min claim area200、丢弃未领取前景。源码 dataclass 的默认值不同，不能将默认值当作最终配方。

证据：
[export_yolo.py L38–54、80–106](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/export_yolo.py#L38-L106)，
[train_detector.py L43–52](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/train_detector.py#L43-L52)，
[fusion.py L84–115、119–151、168–200](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/fusion.py#L84-L200)，
[报告 L105–109](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/reports/main.tex#L105-L109)。

较便宜的实例聚合备选也值得研究：其 `bridge_and_label` 先膨胀找组件归属，再把 group label 映射回原来的前景像素，不把填入的“桥”当成提交边界。这能把“改变实例归属”与“画出不存在的连接像素”分开。证据：
[postprocess.py L110–124](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/postprocess.py#L110-L124)。

## 验证协议：能学习设计，不能横向比较分数

作者报告563张训练观察、144张验证观察（906/248个 annotator view），先按**日**分组，再随机打乱组（seed0，20% dates作验证）。同一日期与同一图的所有 annotators 留在同侧，避免同像素跨侧；但实现**没有27日 embargo，也不是按未来时间整块外推**。作者自己在 data.py 头部承认太阳活动持续多日，并提供更严格的 month grouping 选项。因此不能把 date split 叫作已排除了跨日相关性，也不能直接与我方2020–2022时间外49图holdout比较。

其验证144张再分为固定两半，各72张；每半使用另一半选出的后处理参数进行评估。最终提交参数在全144张上重新选择。训练 checkpoint 的每epoch验证仅用调参半中的48张（train.py默认上限24，报告实际配方48），最后报告改用 final epoch；作者比较多个模型、seed、detector组合，并查看部分 withheld-half 结果，所以这个 cross-fit 对后处理选择的控制不等于对整段研究过程的完全独立外部验证。

`fuse_detections.py` 的默认网格是3×1×3×4×2×3×1=216种fusion设置，并打印调参排名及另一半 top5 的结果。**这不是我方“少量预先定义V4实验”的预算**；不复刻其大网格，更不能用 Public Leaderboard 扫参数。cross-fitting后的 bootstrap 是按 observation 一起重采样其所有 views，而不是把多个 annotator 当成独立图。

证据：
[data.py L125–156](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/data.py#L125-L156)，
[train.py L49–59、99–102、136–147](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/train.py#L49-L147)，
[fuse_detections.py L153–165、184–189、199–254](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/scripts/fuse_detections.py#L153-L254)，
[scoring.py L22–35、48–61](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/scoring.py#L22-L61)。

### 重要评分实现差异

该仓库虽然称遵循官方评价，但 `metrics.py` L83–109 是按 IoU>0.5 候选做 **greedy one-to-one matching**；L152–159 定义 FP=n_pred−TP、FN=n_gt−TP。我方已保存的官方 notebook 则把所有 IoU>0.5 命中对计入 TP，并把无命中的列/行记为 FP/FN。只有双方各自的实例集合互斥、使>0.5匹配唯一时，两者才等价；不能假设所有同 annotator GT polygons 都互斥。

另其 evaluate L238–243 对 GT 字典之外的 prediction-image 加FP，而官方 notebook 是遍历GT组。其 fragmentation 使用对GT面积的覆盖阈值（L112–141），也不等于官方示例的所有非零IoU关系计数。这些差异不代表其全体实测数值必然错误，但说明**不能直接把该repo的scorer替换我方scorer，也不能把公开数值当作与我方官方语义完全可比的基准**。P1仍以主办方 notebook 为准。

证据：[metrics.py L83–109、112–164、227–244](https://github.com/ameyypawar/filament-seg/blob/81a739bb7786a44b989f5d76afbb6f9d22b5d1dd/filament_seg/metrics.py#L83-L244)。

### 公开分数的使用边界

README列出的 U-Net0.371、融合0.421、Public0.37 是**作者的报告值**，本次没有执行或独立复现。它们仅用于说明作者研究过什么，不与我方0.1056的冻结时间外holdout作差、比值或“提升”结论。其 all-data 权重训练包含原验证图，作者明确不赋予其验证分数。对我方而言，下载并使用其已在707张训练图上拟合的权重，还会碰到我方保留holdout的污染问题；本次没有下载这些权重。

## 许可和外部模型

| 材料 | 读到的许可/来源 | 我方动作 |
|---|---|---|
| filament-seg 源码 | MIT，Amey Pawar，2026 | 可借鉴或改编；复制实质源码时保留完整MIT和署名，说明修改 |
| segmentation_models_pytorch | 官方MIT，Pavel Iakubovskii | 可作为公开依赖；记录版本和许可证 |
| torchvision实现 | 官方BSD-3-Clause | 代码许可与其下载的模型权重、ImageNet数据条件分开记录 |
| ResNet34 encoder weights | 源码 `encoder_weights='imagenet'`；本次未下载 | 若后续采用，应记录实际权重URL/SHA、数据来源和许可；不以代码MIT自动覆盖权重许可 |
| 作者训练后权重 | README称 Kaggle dataset CC BY-NC4.0 | 本次不使用；不要与MIT源码混称许可相同 |
| Ultralytics/YOLO11 | 官方上游 LICENSE 为 **AGPL-3.0**；项目 requirements-detector 固定 ultralytics8.4.173 | 仓库MIT不把第三方YOLO代码或权重改成MIT；若采用需履行相应开源/分发义务或具备适当商业许可。AGPL本身并非“禁止商业使用”许可，也不在此无根据断言比赛禁止它 |
| 官方比赛图像及多边形 | 该repo声明MAGFiLO CC BY-NC4.0、只用竞赛提供训练部分 | 我方继续只从官方比赛数据取得训练监督，绝不下载外部全量MAGFiLO标注 |

上游来源：[SMP LICENSE](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE)、
[torchvision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE)、
[Ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。
上游许可证快照不是对固定8.4.173 wheel全部依赖/权重的完整法律审计。

复现时还要注意其 requirements.txt 的说明写“训练/推理 torch2.10.0”，实际 pin 是 torch2.14.0、torchvision0.29.0（L18–22）。这是文件内的可见不一致，不能将其依赖文件无修改搬到我方已验证的Kaggle CUDA环境。直接保留我方remote runtime freeze，按实际新依赖增量记录即可。

## 给P4的最小建议（需等P2诊断确认）

1. 保留v3原始分数，不重训、不覆盖、不重评其holdout。采用我方开发日期分组及27日 embargo；源码和预算在首个GPU任务前冻结。
2. 第一项只修复**原生512 crop + 前景/背景明确采样 + 原分辨率mask + tiled inference**，尽量保留现小U-Net和损失，隔离表示尺度/采样问题；不要同时加ResNet、预训练、YOLO和大规模TTA。
3. 若P2显示盘内径向光度漂移，第二项固定加入limb correction和radius通道；与第一项同日期划分、seed、更新步数和评价规则。
4. 从同一固定OOF概率图比较最多两个预定义聚合策略（原连接域；小间隙仅归并实例、边界仍取原前景），报告FP/FN和失败图，不做后处理网格或排行榜扫描。
5. 只有像素召回/边界已可用，而主要损失仍是碎片化或过度合并，才值得后续训练一个检测器。先训练三套60–80epoch YOLO并组合不符合本轮“少量实验”，而且还增加许可和算力成本。ResNet34也应作为后续独立架构因素，不能用它掩盖前两项是否修复了明确根因。

以上是由源码支持的候选设计建议，不是已经实现或完成的V4实验，也不预言其PQ会提升。
