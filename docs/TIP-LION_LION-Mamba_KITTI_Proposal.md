TIP-LION

## 面向 LION-Mamba 的拓扑信息保持序列化与信息增 益体素生成

一个聚焦 KITTI 的点云 3D 检测论文级方法方案

核心命题。 LION-Mamba 的真正瓶颈不是 “是否使用 Mamba”，而是将稀疏 3D 体素强制压平成 1D 序列时，哪些 3D 邻接、局部流形、密度和前景边界信息被提前丢失。TIP-LION 将该问题重写为一个图到序列的信息保持问题: 先在体素层构建度量测度图，再用拓扑带宽约束的序列化、热核几何描述子和信息增益体素生成，尽可能在进入 Mamba 之前保留 3D 可检测信息。

数据集 KITTI 3D Object Detection，三类目标：Car / Pedestrian / Cyclist。

基线

已复现的 LION-Mamba，保持 SECOND 风格 BEV backbone 与 detection head。

优先级可解释性纯点云侧方法。图像融合不作为本文主线。

以“序列撕裂率、局部信息保留率、热核响应、生成体素信息增益”为机制诊断指标。

<!-- Meanless: Version: 2026-05-26 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 目录

1 问题重述: LION-Mamba 中真正值得攻击的瓶颈 2

1.1 为什么 KITTI 上这个问题依然成立 2

2 方法总览 2

3 理论动机：把序列化看成信息保持映射 3

3.1 从 Mamba 的状态传播看 1D 序列化损失 3

3.2 为什么原始 LION 的 3D SFD 仍可继续深化 3

4 模块一：TIS 拓扑信息保持序列化 3

4.1 构建局部度量测度图 3

4.2 带宽约束序列化 4

4.3 与 Mamba 的耦合方式 4

5 模块二：HGD 热核几何描述子 4

5.1 热核返回概率：局部维度与密度的统一描述 5

5.2 局部协方差谱与方向熵 5

5.3 注入方式：调制输入特征与状态空间参数 5

6 模块三：IGVG 信息增益体素生成 6

6.1 候选体素的信息增益评分 6

6.2 热扩散初始化而非零初始化或 KNN 复制 6

7 与现有 LION-Mamba 代码的对应关系 6

8 人工模拟数据自审与可解释性 7

8.1 合成设置 7

8.2 IGVG 的可解释性视图 8

9 KITTI 实验设计：论文级验证而不是开关实验 8

9.1 主实验 8

9.2 机制诊断指标 8

9.3 预期最有说服力的结果形态 9

10 论文贡献表述 9

11 风险边界与应对 9

12 结论 10

<!-- Meanless: 1/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 1 问题重述：LION-Mamba 中真正值得攻击的瓶颈

LION 的贡献在于用线性 RNN/Mamba 类算子在大 group 内完成长程特征交互, 并通过 3D spatial feature descriptor 与 voxel generation 补偿稀疏点云中的空间建模缺陷。原论文已经明确指出：线性 RNN 需要顺序输入，体素被压平成 1D 序列后，相邻 3D 体素可能在序列上相距很远，从而导致 3D 空间信息损失；同时 voxel generation 通过前景高响应区域与自回归生成能力缓解稀疏问题 [1]。

但从论文级创新角度看，仅继续增加扫描方向、替换排序规则或加若干手工几何特征，问题不够深。更本质的问题是：

核心问题。给定稀疏体素集合 $\mathcal{V} = {\left\{  {v}_{i}\right\}  }_{i = 1}^{L}$ 及其 3D 坐标、点密度和语义响应，如何构造一个序列化映射

$$
\pi  : \mathcal{V} \rightarrow  \{ 1,2,\ldots ,L\}
$$

使得 Mamba 在 1D 序列上的状态传播尽可能近似 3D 度量空间中的信息传播？

这将 LION-Mamba 的难点从 “设计一个新 scan order” 提升为 “在 3D 到 1D 的压缩映射中最大化保留对象检测所需信息”。因此，本文方案不把 3D spatial feature descriptor 视作一个普通局部卷积补丁，而是把它扩展为一个预序列化信息保持算子族。

### 1.1 为什么 KITTI 上这个问题依然成立

KITTI 的样本规模相对小，场景前向、点云稀疏、Pedestrian/Cyclist 目标细长且远距离点数不足。LION 在 KITTI 中使用大 group 与 $x/y$ 两个方向序列化可以建立长程依赖，但这并不保证局部几何在序列中被保留。对于 Pedestrian、Cyclist 这类小目标，局部形状边界、垂直结构、轮廓连续性和局部密度变化比纯长程建模更敏感。因此，KITTI 不只是 “能跑得动” 的小数据集，而是检验 3D 到 1D 信息瓶颈的合适平台。

## 2 方法总览

本文方法命名为 TIP-LION: Topology-Information-Preserving LION. 它由三个彼此耦合、但可以独立消融的模块组成：

1. TIS: Topology-Information Serialization, 拓扑信息保持序列化。将体素看成局部度量图，显式最小化 3D 邻接边在 1D 序列中的撕裂。

2. HGD: Heat-kernel Geometry Descriptor, 热核几何描述子。在进入 Mamba 前, 用热扩散、局部协方差谱和方向熵提取 3D 局部流形信息，替代 “只靠卷积补偿” 的描述子。

3. IGVG: Information-Gain Voxel Generation, 信息增益体素生成。将 LION 中固定 offset 的前景扩散改写为 “预期信息增益最大” 的虚拟体素生成。

<!-- Media -->

<!-- figureText: Sparse voxels<br>Metric-measure graph $\mathcal{G}$<br>HGD geometry code<br>TIS serialization<br>LION-Mamba large group RNN<br>BEV backbone detection head<br>3D 邻接、密度<br>range、响应<br>减少 3D 边在 1D 中断裂<br>IGVG virtual voxels<br>只在信息缺口处补体素 -->

<img src="https://cdn.noedgeai.com/bo_d8apnrs91nqc73c7mjig_2.jpg?x=174&y=1787&w=1304&h=254&r=0"/>

<!-- Media -->

图 1: TIP-LION 的整体框架。它不是简单替换 Mamba，而是在 Mamba 前显式处理 3D 到 1D 的信息保持问题。

<!-- Meanless: 2/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 3 理论动机：把序列化看成信息保持映射

### 3.1 从 Mamba 的状态传播看 1D 序列化损失

Mamba/线性 RNN 在序列中传播信息时，隐状态对相距 $\Delta$ 的 token 通常具有随距离衰减的有效耦合。抽象地，可以用一个非增核函数 $k\left( \Delta \right)$ 表示序列邻近性：

$$
{K}_{\pi }\left( {i,j}\right)  = k\left( \left| {\pi \left( i\right)  - \pi \left( j\right) }\right| \right) ,\;{k}^{\prime }\left( \Delta \right)  \leq  0.
$$

其中 $\pi$ 是 3D 体素到 1D 序列位置的映射。另一方面，点云中的对象几何可由一个局部图 $\mathcal{G} = \; \left( {\mathcal{V},\mathcal{E},W}\right)$ 表示，其中 ${W}_{ij}$ 表示 ${v}_{i}$ 与 ${v}_{j}$ 在 3D 空间、密度和前景响应上的邻接强度。理想情况是： 若 ${W}_{ij}$ 很大,则 ${K}_{\pi }\left( {i,j}\right)$ 也应较大。

因此定义一个序列化信息保持代理量：

$$
\mathcal{I}\left( \pi \right)  = \frac{\mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{W}_{ij}k\left( \left| {\pi \left( i\right)  - \pi \left( j\right) }\right| \right) }{\mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{W}_{ij}}. \tag{1}
$$

对应的信息损失为：

$$
{\mathcal{L}}_{\text{ tear }}\left( \pi \right)  = \mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{W}_{ij}\left\lbrack  {1 - k\left( \left| {\pi \left( i\right)  - \pi \left( j\right) }\right| \right) }\right\rbrack  . \tag{2}
$$

这相当于图的 bandwidth / linear arrangement 问题: 将一个 3D 邻接图排成一条线时，尽量避免强邻接边被拉得过长。这个视角比 “再加一种 scan order” 更根本，因为它给出了可优化、 可解释、可度量的目标。

### 3.2 为什么原始 LION 的 3D SFD 仍可继续深化

原始 LION 的 3D spatial feature descriptor 通过 3D submanifold convolution 提供局部空间补偿，并且消融显示其对 LION 有明显贡献 [1]。但它并没有直接约束序列化映射 $\pi$ ，也没有显式度量哪些 3D 邻接在 1D 中被破坏。TIP-LION 的立场是：

3D SFD 不应只是 “卷积补偿器”，而应成为预序列化信息保持层：一方面为每个 token 注入局部流形与密度结构，另一方面引导 token 排序，使 Mamba 的 1D 状态传播更接近 3D 图上的信息扩散。

## 4 模块一：TIS 拓扑信息保持序列化

### 4.1 构建局部度量测度图

在每个 LION group 或 sparse window 内, 令体素 token 为

$$
{v}_{i} = \left( {{\mathbf{p}}_{i},{\mathbf{f}}_{i},{n}_{i},{r}_{i}}\right) ,
$$

其中 ${\mathbf{p}}_{i} \in  {\mathbb{R}}^{3}$ 是体素中心坐标, ${\mathbf{f}}_{i}$ 是 VFE 输出特征, ${n}_{i}$ 是 voxel 内点数, ${r}_{i}$ 是到 LiDAR 原点的 range。定义 range-normalized density:

$$
{\rho }_{i} = \log \left( {1 + {n}_{i}}\right)  \cdot  \left( {1 + \alpha {r}_{i}^{2}}\right) . \tag{3}
$$

该项用于部分抵消远距离体素天然更稀疏的问题。

<!-- Meanless: 3/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

对局部邻接边 $\left( {i,j}\right)$ ,定义权重:

$$
{W}_{ij} = \exp \left( {-\frac{{\begin{Vmatrix}{\mathbf{p}}_{i} - {\mathbf{p}}_{j}\end{Vmatrix}}_{2}^{2}}{{\sigma }_{p}^{2}}}\right) \exp \left( {-\frac{\left| {\rho }_{i} - {\rho }_{j}\right| }{{\sigma }_{\rho }}}\right) \left( {1 + \gamma {q}_{i}{q}_{j}}\right) , \tag{4}
$$

其中 ${q}_{i}$ 是 detach 后的前景响应或 feature response。 ${q}_{i}{q}_{j}$ 不是监督标签，而是弱前景一致性项，用来让对象区域内部的邻接边获得更高优先级。

### 4.2 带宽约束序列化

TIS 的目标是找到 $\pi$ ，使式 (2) 尽可能小。精确求解 minimum linear arrangement 代价较高， 因此采用低成本近似:

TIS 近似策略。

1. 由 $W$ 构造稀疏邻接矩阵 $A = \mathbb{1}\left( {W > \epsilon }\right)$ 。

2. 使用带宽约束排序，例如 Reverse Cuthill-McKee、Fiedler 向量排序或数步热扩散坐标排序，得到拓扑顺序 ${\pi }_{\mathrm{{tis}}}$ 。

3. 与 LION 原有 $x/y$ 两个方向形成互补 order:

$$
\Pi  = \left\{  {{\pi }_{x},{\pi }_{y},{\pi }_{\mathrm{{tis}}}}\right\}  .
$$

4. 采用 channel grouping: 不同通道组走不同 order，再通过轻量 gate 汇合。

这里不建议把 order 数量无限加大。TIS 的价值在于第三个 order 不是手工轴向扫描，而是由局部 3D 图的带宽目标推导出来。它对应 “哪些 3D 边最不能被撕裂”的明确假设。

### 4.3 与 Mamba 的耦合方式

给定第 $g$ 个通道组的输入 ${X}^{\left( g\right) } \in  {\mathbb{R}}^{L \times  {C}_{g}}$ ，按 order ${\pi }_{g}$ 重排：

$$
{\widetilde{X}}^{\left( g\right) } = {P}_{{\pi }_{g}}{X}^{\left( g\right) }.
$$

经过 Mamba/LION operator 后再逆映射回原始 sparse voxel index:

$$
{Y}^{\left( g\right) } = {P}_{{\pi }_{g}}^{-1}\operatorname{Mamba}\left( {\widetilde{X}}^{\left( g\right) }\right) .
$$

最终输出为：

$$
Y = \mathop{\sum }\limits_{g}{\omega }_{g}\left( \mathbf{z}\right) {Y}^{\left( g\right) },\;\mathop{\sum }\limits_{g}{\omega }_{g} = 1, \tag{5}
$$

其中 ${\omega }_{g}$ 可由全局 density/range 统计或 group feature mean 产生。若追求实现稳定，也可以先使用固定平均融合。

## 5 模块二：HGD 热核几何描述子

TIS 解决 “如何排序”，HGD 解决“排序前 token 自身应携带什么 3D 信息”。它的目标不是再堆一个 3D conv，而是用图扩散与局部谱几何刻画体素的局部流形结构。

<!-- Meanless: 4/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

### 5.1 热核返回概率：局部维度与密度的统一描述

给定图拉普拉斯 $L = D - W$ ,热扩散算子为:

$$
{H}_{t} = \exp \left( {-{tL}}\right) .
$$

热核对角项

$$
{h}_{i}\left( t\right)  = {\left\lbrack  \exp \left( -tL\right) \right\rbrack  }_{ii} \tag{6}
$$

可以解释为从 ${v}_{i}$ 出发的热量在时间 $t$ 后返回自身的概率。直观上，曲面边界、细长结构、稀疏断裂处会呈现不同的多尺度热核响应。实际实现中不需要显式矩阵指数，可用随机游走或 Chebyshev 多项式近似：

$$
\exp \left( {-{tL}}\right) \mathbf{x} \approx  \mathop{\sum }\limits_{{m = 0}}^{M}{c}_{m}\left( t\right) {T}_{m}\left( \widetilde{L}\right) \mathbf{x}.
$$

对于 KITTI，使用 $M = 3 \sim  5$ 的近似已经足够作为描述子，而不是精确谱分解。

### 5.2 局部协方差谱与方向熵

对体素 ${v}_{i}$ 的邻域 $\mathcal{N}\left( i\right)$ ，计算加权协方差：

$$
{\sum }_{i} = \frac{1}{{Z}_{i}}\mathop{\sum }\limits_{{j \in  \mathcal{N}\left( i\right) }}{W}_{ij}\left( {{\mathbf{p}}_{j} - {\overline{\mathbf{p}}}_{i}}\right) {\left( {\mathbf{p}}_{j} - {\overline{\mathbf{p}}}_{i}\right) }^{\top }.
$$

令特征值为 ${\lambda }_{1} \geq  {\lambda }_{2} \geq  {\lambda }_{3}$ ，构造常用局部几何不变量：

$$
{\ell }_{i} = \frac{{\lambda }_{1} - {\lambda }_{2}}{{\lambda }_{1} + \epsilon }
$$

$$
\text{ linearity, } \tag{7}
$$

$$
{p}_{i} = \frac{{\lambda }_{2} - {\lambda }_{3}}{{\lambda }_{1} + \epsilon }
$$

$$
\text{ planarity, } \tag{8}
$$

$$
{s}_{i} = \frac{{\lambda }_{3}}{{\lambda }_{1} + \epsilon }
$$

scattering.(9)

再定义方向熵：

$$
{E}_{i} =  - \mathop{\sum }\limits_{{b = 1}}^{B}{a}_{ib}\log \left( {{a}_{ib} + \epsilon }\right) , \tag{10}
$$

其中 ${a}_{ib}$ 是邻居方向落入第 $b$ 个角度 bin 的归一化权重。方向熵低的地方往往是边界、遮挡切面或细长结构, Pedestrian/Cyclist 尤其敏感。

### 5.3 注入方式：调制输入特征与状态空间参数

将 HGD 特征拼接为

$$
{\mathbf{d}}_{i} = \left\lbrack  {{h}_{i}\left( {t}_{1}\right) ,\ldots ,{h}_{i}\left( {t}_{S}\right) ,{\ell }_{i},{p}_{i},{s}_{i},{E}_{i},{\rho }_{i},{r}_{i}}\right\rbrack  .
$$

通过小 MLP 得到门控：

$$
{\mathbf{g}}_{i} = \sigma \left( {\operatorname{MLP}\left( {\mathbf{d}}_{i}\right) }\right) ,\;{\widehat{\mathbf{f}}}_{i} = {\mathbf{f}}_{i} + {\mathbf{g}}_{i} \odot  \phi \left( {\mathbf{d}}_{i}\right) . \tag{11}
$$

更进一步，可将 ${\mathbf{d}}_{i}$ 用于调制 Mamba 的选择性参数，例如步长或输入门：

$$
{\Delta }_{i}^{\prime } = {\Delta }_{i} + {\mathrm{{MLP}}}_{\Delta }\left( {\mathbf{d}}_{i}\right) . \tag{12}
$$

<!-- Meanless: 5/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

这使 Mamba 在稀疏边界和远距离稀疏区具备不同的信息传播节奏，而不是把所有 token 当成同质序列元素。

## 6 模块三：IGVG 信息增益体素生成

原始 LION 的 voxel generation 先选高响应体素，再按固定对角 offset 扩散，最后用自回归 block 生成新体素特征 [1]。该设计简洁，但隐含两个强假设：第一，高响应区域一定是最佳生成起点；第二，固定 offset 足以覆盖有用的缺失邻域。

TIP-LION 将生成体素的标准改为：一个候选虚拟体素是否能够最大化降低 3D 图到 1D 序列的信息缺口。

### 6.1 候选体素的信息增益评分

对候选虚拟体素 $c$ ，定义其潜在邻接集合 $\mathcal{N}\left( c\right)$ ，并估计插入 $c$ 后的信息保持变化：(13)

<!-- Media -->

<!-- figureText: $S\left( c\right) = \underset{\text{ foreground support }}{\underbrace{\mathop{\sum }\limits_{{j \in \mathcal{N}\left( c\right) }}{\widetilde{W}}_{cj}{q}_{j}}} \cdot \underset{\text{ boundary / occlusion }}{\underbrace{\left( 1 + \eta {E}_{j}\right) }} \cdot \underset{\text{ sparse gap }}{\underbrace{\left( 1 - {\overline{\rho }}_{c}\right) }} \cdot \underset{\text{ serialization gain }}{\underbrace{\Delta \mathcal{I}\left( c\right) }}$ -->

<img src="https://cdn.noedgeai.com/bo_d8apnrs91nqc73c7mjig_6.jpg?x=345&y=757&w=951&h=146&r=0"/>

<!-- Media -->

其中 $\Delta \mathcal{I}\left( c\right)$ 是将 $c$ 插入当前图与序列后，式 (1) 的预期提升。直观上，IGVG 不再固定向四个对角扩散，而是在“前景支持高、局部稀疏、边界熵异常、且有助于减少序列撕裂”的位置生成体素。

### 6.2 热扩散初始化而非零初始化或 KNN 复制

原始 LION 的零初始化依赖后续自回归 block 生成特征。TIP-LION 保留自回归生成能力，但加入热扩散初始化:

$$
{\mathbf{f}}_{c}^{0} = \frac{\mathop{\sum }\limits_{{j \in  \mathcal{N}\left( c\right) }}{\widetilde{W}}_{cj}{\mathbf{f}}_{j}}{\mathop{\sum }\limits_{{j \in  \mathcal{N}\left( c\right) }}{\widetilde{W}}_{cj} + \epsilon },\;{\widehat{\mathbf{f}}}_{c}^{0} = {\alpha }_{c}{\mathbf{f}}_{c}^{0}, \tag{14}
$$

其中 ${\alpha }_{c} = \sigma \left( {\operatorname{MLP}\left( \left\lbrack  {S\left( c\right) ,{\rho }_{c},{E}_{c}}\right\rbrack  \right) }\right)$ 是置信门控。这样可以避免把错误生成体素以高置信度注入 backbone.

## 7 与现有 LION-Mamba 代码的对应关系

<!-- Media -->

<table><tr><td>模块</td><td>插入点</td><td>核心改动</td></tr><tr><td>TIS</td><td>序列化映射；LION 层</td><td>在原 $x/y$ order 外加入图带宽约束 order; 使用 channel group 或 fixed average 融合。</td></tr><tr><td>HGD</td><td>VFE 输出处，LION block 输入处</td><td>利用 voxel coordinates、voxel count、range 与 feature response 构建局部图描述子；投影后注入 token feature。</td></tr><tr><td>IGVG</td><td>PatchMerging3D; diffusion 逻辑</td><td>将固定 offset diffusion 替换为信息增益候选筛选; 生成体素用热扩散初始化加 confidence gate。</td></tr><tr><td>诊断指标</td><td>train / eval hooks</td><td>记录 $\mathcal{I}\left( \pi \right)$ 、tear rate、虚拟体素保留率、距离分段 recall、 类别分段 AP。</td></tr></table>

表 1: TIP-LION 在 LION 源码中的主要接口。该方案保留检测头与 BEV backbone，把创新集中在 3D backbone 的预序列化与体素生成。

<!-- Media -->

<!-- Meanless: 6/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 8 人工模拟数据自审与可解释性

在正式 KITTI 实验之前，可以用合成稀疏体素图进行机制级自审。这里的目的不是证明最终 AP， 而是检查方法是否真的减少了 3D 到 1D 的几何撕裂。

### 8.1 合成设置

构造一个由曲面、遮挡缺口、细长竖直结构和远距离稀疏结构组成的 2D 投影体素图，并用局部距离、密度相似性建立图边。比较三种序列化：轴向 order、Morton/Z-order、TIP-LION 的图带宽约束 order。评价指标包括:

- retained local information: 式 (1); 越高越好。

- weighted tear rate: 高权重 3D 邻接边在 1D 中被拉远的比例；越低越好。

- successor distance: 序列相邻 token 的平均 3D 距离；越低通常代表局部连续性越好。

<!-- Media -->

Synthetic serialization tearing: 3D neighborhood versus 1D sequence

<!-- figureText: Axis x/y order<br>Morton / Z-order<br>TIP-LION geodesic order<br>。<br>2023-03-03-03<br>___ -->

<img src="https://cdn.noedgeai.com/bo_d8apnrs91nqc73c7mjig_7.jpg?x=184&y=794&w=1286&h=434&r=0"/>

图 2: 合成数据上的序列撕裂可视化。灰色线表示序列相邻 token，红色线表示较大的 3D 跳跃。TIP-LION 的图带宽约束 order 更倾向沿局部几何连通结构展开。

<!-- figureText: Toy self-check: topology-preserving order improves the information surrogate<br>1.0<br>retained local information<br>weighted tear rate<br>0.82<br>0.8<br>0.70<br>normalized score<br>0.6<br>0.53<br>0.4<br>0.2<br>0.05<br>0.08<br>0.00<br>0.0<br>Axis<br>Morton<br>TIP-LION -->

<img src="https://cdn.noedgeai.com/bo_d8apnrs91nqc73c7mjig_7.jpg?x=293&y=1398&w=1064&h=506&r=0"/>

图 3: 合成自审指标。一次模拟中, 轴向 order 的 retained local information 为 0.53 , Morton 为 0.70, TIP-LION 为 0.82; TIP-LION 的 weighted tear rate 为 0.00。该结果只用于机制说明，不应被解读为 KITTI AP 预期。

<!-- Media -->

<!-- Meanless: 7/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

### 8.2 IGVG 的可解释性视图

IGVG 还可以可视化 “为什么生成这些体素”：固定 offset diffusion 可能在高响应 seed 周围机械复制，而信息增益生成会更偏向边界、稀疏缺口和形状连续性断裂处。

<!-- Media -->

Voxel densification as expected information gain rather than fixed offset cloning

<!-- figureText: Fixed diagonal diffusion<br>Information-gain voxel generation<br>observed voxel<br>8<br>generated voxel<br>助康康保险环保环境保养<br>2023年12月<br>___<br>___<br>2022年11月11日<br>___.<br>圆<br>20 2018年11月27日<br>....... 10/10/10/10/10/10<br>......................<br>___。<br>___. -->

<img src="https://cdn.noedgeai.com/bo_d8apnrs91nqc73c7mjig_8.jpg?x=255&y=461&w=1140&h=288&r=0"/>

图 4: 固定 offset 体素扩散与信息增益体素生成的对比示意。蓝色为观测体素，金色为前景 seed，彩色方块为生成体素。IGVG 的生成位置由局部稀疏性、边界熵和预期信息增益共同决定。

<!-- Media -->

## 9 KITTI 实验设计：论文级验证而不是开关实验

### 9.1 主实验

在已复现 LION-Mamba 的基础上, 所有实验保持数据划分、训练 epoch、优化器、BEV backbone 和 detection head 一致。主表不应只报 overall AP，而应至少包含：Car / Pedestrian / Cyclist 的 Easy / Moderate / Hard 3D AP 与 BEV AP，并额外提供按距离段和遮挡等级的分组结果。

<!-- Media -->

<table><tr><td>实验项</td><td>目的</td></tr><tr><td></td><td>固定已复现基线。</td></tr><tr><td>LION-Mamba baseline</td><td></td></tr><tr><td>Baseline + HGD</td><td>验证热核/谱几何描述子是否优于原始局部空间补偿。</td></tr><tr><td>Baseline + TIS</td><td>验证拓扑带宽约束 order 是否减少 3D-1D 撕裂并提升小目标。</td></tr><tr><td>Baseline + IGVG</td><td>验证信息增益体素生成是否优于固定 offset 扩散。</td></tr><tr><td>HGD + TIS</td><td>验证“描述子”和“排序”是否互补。</td></tr><tr><td>Full TIP-LION</td><td>验证完整方案。</td></tr></table>

表 2：主实验矩阵。这里的消融不是为了快速试错，而是为了证明三条机制链：局部几何、序列保真、稀疏补全。

<!-- Media -->

### 9.2 机制诊断指标

1. 序列撕裂率。统计强 3D 邻接边 $\left( {i,j}\right)$ 中满足 $\left| {\pi \left( i\right)  - \pi \left( j\right) }\right|  > \tau$ 的比例，并按类别 foreground 区域分别统计。

2. 信息保留率。使用式 (1)，比较 $x/y$ 、Morton、TIS order。

<!-- Meanless: 8/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

3. HGD 可视化。将热核响应、局部线性度、方向熵投影到 BEV，看高响应区域是否对应 Pedestrian/Cyclist 的边界、遮挡切面和远距离稀疏结构。

4. IGVG 生成质量。统计生成体素与 GT box 的 overlap、生成体素在前景/背景中的比例、生成后中远距离 recall 的变化。

5. 错误案例分解。分析 false negative 是否集中在远距离小目标、遮挡目标或边界断裂目标。

### 9.3 预期最有说服力的结果形态

理想结果不是 Car AP 大幅提升, 而是: Pedestrian / Cyclist moderate 与 hard 有稳定提升; 远距离和遮挡分组 recall 上升；TIS 的信息保留率与 AP 提升存在正相关；IGVG 生成体素主要落在 GT box 或其局部边界附近，而不是在背景中无意义扩散。

## 10 论文贡献表述

可以将论文主线写成: Mamba-based 3D detectors suffer from a pre-serialization information bottleneck. We formulate voxel serialization as a topology-information preservation problem and propose graph-bandwidth-aware serialization, heat-kernel geometry descriptors, and information-gain voxel generation to preserve 3D structure before sequential state-space modeling.

对应中文贡献可表述为：

1. 首次从图到序列的信息保持角度分析 LION-Mamba 的 3D 到 1D 序列化瓶颈，并提出可量化的序列信息保留目标。

2. 提出拓扑信息保持序列化 TIS，将体素局部度量图的带宽约束引入 LION-Mamba 的 token ordering, 减少强 3D 邻接边在 1D 序列中的撕裂。

3. 提出热核几何描述子 HGD，用多尺度热扩散、局部协方差谱和方向熵描述稀疏体素的局部流形结构，使 Mamba token 在进入序列前携带可解释几何信息。

4. 提出信息增益体素生成 IGVG，将 LION 的固定 offset foreground diffusion 扩展为由局部稀疏性、边界熵和序列信息增益驱动的自适应体素补全。

5. 在 KITTI 上通过 AP、距离分段、遮挡分段、序列撕裂率和生成体素可视化，验证方法对小目标与稀疏远距离目标的有效性。

## 11 风险边界与应对

风险 1: 图排序带来额外开销。不应对每个大 group 做昂贵特征分解。优先使用稀疏图 bandwidth heuristic、局部 RCM、少步热扩散坐标或离散近似。训练阶段可以 detach order， 避免反向传播穿过排序。

风险 2: 生成体素引入背景噪声。IGVG 必须加入 confidence gate 和生成比例上限。论文中要展示生成体素的前景比例、背景误扩散案例和生成前后 recall 变化。

风险 3: KITTI 数据量小, 容易过拟合。方法设计应尽量参数轻量, 重点强调结构性 inductive bias 与机制指标，而不是大规模参数增长。

风险 4: TIS 与 HGD 贡献混淆。消融必须拆开：只改描述子、只改排序、只改生成、两两组合、完整组合。否则很难说清方法有效性来自哪里。

<!-- Meanless: 9/11 -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 12 结论

TIP-LION 的核心不是 “在 LION-Mamba 上加一个模块”，而是把 LION-Mamba 的根问题重新定义为：稀疏 3D 体素在进入 1D 状态空间模型之前，如何最大限度保留对象检测所需的拓扑、 局部流形、密度和前景边界信息。

这一路线与 LION 原论文的两个观察直接相接：线性 RNN 适合长程建模，但缺乏原生 3D 空间建模；体素生成能缓解稀疏性，但固定扩散策略仍然粗糙。TIP-LION 在这两个观察之上进一步给出信息论和图几何层面的统一解释，并形成可实现、可消融、可解释的完整论文方案。

<!-- Meanless: ${10}/{11}$ -->


<!-- Meanless: TIP-LION proposal<br>KITTI / LION-Mamba -->

## 参考文献

[1] Z. Liu, J. Hou, X. Wang, X. Ye, J. Wang, H. Zhao, and X. Bai. LION: Linear Group RNN for 3D Object Detection in Point Clouds. arXiv:2407.18232, 2024.

[2] A. Gu and T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752, 2023.

[3] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? The KITTI Vision Benchmark Suite. CVPR, 2012.

[4] Y. Yan, Y. Mao, and B. Li. SECOND: Sparsely Embedded Convolutional Detection. Sensors, 2018.

[5] H. Wang et al. DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets. CVPR, 2023.

[6] M. Fiedler. Algebraic connectivity of graphs. Czechoslovak Mathematical Journal, 1973.

[7] J. Sun, M. Ovsjanikov, and L. Guibas. A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion. SGP, 2009.

[8] Y. Zhou and O. Tuzel. VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. CVPR, 2018.

<!-- Meanless: ${11}/{11}$ -->
