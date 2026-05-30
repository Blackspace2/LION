<!-- Meanless: arXiv:2406.10700v2 [cs.CV] 18 Jun 2024 -->

# Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection

Guowen Zhang ${}^{1,2}$ , Lue Fan ${}^{3}$ , Chenhang He ${}^{1}$ , Zhen Lei ${}^{2,3}$ , Zhaoxiang Zhang ${}^{2,3, * }$ , Lei Zhang ${}^{1, * } \; {}^{1}$ The Hong Kong Polytechnic University

${}^{2}$ Center for Artificial Intelligence and Robotics,HKISI,CAS

${}^{3}$ Institute of Automation,Chinese Academy of Sciences

guowen.zhang@connect.polyu.hk, \{csche, cslzhang\}@comp.polyu.edu.hk

\{fanlue2019, zlei, zhaoxiang.zhang\}@ia.ac.cn

Code: https://github.com/gwenzhang/Voxel-Mamba

## Abstract

Serialization-based methods, which serialize the 3D voxels and group them into multiple sequences before inputting to Transformers, have demonstrated their effectiveness in 3D object detection. However, serializing 3D voxels into 1D sequences will inevitably sacrifice the voxel spatial proximity. Such an issue is hard to be addressed by enlarging the group size with existing serialization-based methods due to the quadratic complexity of Transformers with feature sizes. Inspired by the recent advances of state space models (SSMs), we present a Voxel SSM, termed as Voxel Mamba, which employs a group-free strategy to serialize the whole space of voxels into a single sequence. The linear complexity of SSMs encourages our group-free design, alleviating the loss of spatial proximity of voxels. To further enhance the spatial proximity, we propose a Dual-scale SSM Block to establish a hierarchical structure, enabling a larger receptive field in the 1D serialization curve, as well as more complete local regions in 3D space. Moreover, we implicitly apply window partition under the group-free framework by positional encoding, which further enhances spatial proximity by encoding voxel positional information. Our experiments on Waymo Open Dataset and nuScenes dataset show that Voxel Mamba not only achieves higher accuracy than state-of-the-art methods, but also demonstrates significant advantages in computational efficiency.

## 1 Introduction

LiDAR-based 3D object detection from point clouds plays an important role in applications of autonomous driving [19, 4], virtual reality [46], and robots [44]. The sparsely, unevenly and irregularly distributed point cloud data make the efficient and effective 3D object detection a very challenging task. To address these long-standing challenges, researchers have recently proposed several strategies to improve the model architecture. One strategy is to switch from PointNet-based models [48, 77, 55] to sparse convolutional neural network (SpCNN)-based models [69, 53, 10, 14, 54, 26] in order for more effective feature extraction. However, the sparse convolution is unfriendly for deployment and optimization, requiring tremendous engineering efforts. Therefore, another strategy is to switch from SpCNN to serialization-based Transformers to address this issue [13, 64, 37, 68, 41]. These methods usually group non-empty 3D voxels into multiple short sequences by serialization techniques such as window partition [64, 13, 37], Z-shape sorting [65], and Hilbert sorting [68], as shown in Figs. 1 (a) and (b), where a sequence is a group of voxels to be processed by Transformer layers.

---

<!-- Footnote -->

*Corresponding Author

<!-- Footnote -->

---

<!-- Meanless: Preprint -->


<!-- Media -->

<!-- figureText: -<br>Voxel Sets<br>Voxel Sets<br>...<br>One Single Sequence<br>(a) Window-based Grouping<br>(b) Curve-based Grouping<br>(c) Voxel Mamba -->

<img src="https://cdn.noedgeai.com/bo_d8buj02lb0pc73eb040g_1.jpg?x=426&y=210&w=947&h=405&r=0"/>

Figure 1: Comparison between (a) window-based grouping, (b) curve-based grouping, and (c) our proposed single group modeling by Voxel Mamba.

<!-- Media -->

However, the serialization of voxels will inevitably sacrifice their spatial proximity. Some neighboring voxels can be far apart from each other after serialization, as illustrated by the two red points in Fig. 1 (b). Such a loss of proximity is difficult to be addressed in the existing serialization methods [68, 65, 37, 64, 13] because the group size is limited by the quadratic complexity of Transformers. This issue becomes even worse when neighboring voxels are grouped into different groups. Inspired by the recent success of State Space Models (SSMs) [21, 20, 58, 18, 80, 35] in language and vision, in this work we propose a simple yet effective group-free design to address the loss of proximity. Specifically, we introduce a Voxel SSM, termed as Voxel Mamba, for 3D object detection from point cloud. The linear computational complexity of SSMs makes it feasible to treat all voxels as a single group and sort them into a single sequence. This results in a group-free modeling of voxels, which is more efficient and deployment-friendly than previous methods since no padding tokens are needed. Nonetheless, even we can sort all voxels into a group-free sequence, it cannot be ensured that all of them are within an effective receptive field.

To enhance the spatial proximity of Voxel Mamba, we further propose two modules with it. The first is the Dual-scale SSM Block (DSB) by introducing the downsampling operations in SSMs. In specific, the forward SSM branches process the high-resolution voxel features, while the backward branches extract features from the low-resolution representation. In this way, we integrate the hierarchical design with the bidirectional design in a more economical way. More importantly, the hierarchy brings a larger effective receptive field for the serialized sequence so that the spatial proximity in local 3D regions can be enhanced. The second module we introduced is the Implicit Window Partition (IWP). The window partition is a widely used strategy in previous methods [13, 64] to enhance the proximity of voxels inside a window. However, it impedes the proximity of voxels across windows and contradicts with our group-free principle. We therefore propose an implicit window partition scheme to embrace its strengths while discarding its weaknesses. In specific, we encode the voxel positions inside and across windows into embeddings for feature learning without explicitly conducting spatial window partition. In this way, better voxel proximity can be achieved under our group-free design with minimal computational cost.

Our contributions are summarized as follows:

- We propose Voxel Mamba, a group-free backbone for voxel-based 3D detection. Voxel Mamba abandons the grouping operation and serializes voxels into one single sequence, enabling better efficiency.

- To mitigate the loss of spatial proximity due to serialization, we propose the Dual-scale SSM Block (DSB) and the Implicit Window Partition (IWP) to enhance the spatial proximity preservation of Voxel Mamba.

- Our method achieves superior performance to previous state-of-the-art methods on the large-scale Waymo Open dataset [59] and nuScenes [2] datasets.

<!-- Meanless: 2 -->


## 2 Related Work

3D Object Detection from Point Clouds. There are two major point cloud representations for 3D object detection, i.e., point-based and voxel-based ones. As in PointNet [49, 50], point-based methods [48, 47, 55, 51, 36] directly extract geometric features from small regions of raw points. However, those methods suffer from low inference efficiency and limited context features. Voxel-based methods [53, 54, 69, 26, 77, 15-17] convert raw points into regular grids through voxelization and then process them with sparse convolution [69] or Transformers [13, 25, 64]. Voxel-based methods are currently the main stream for 3D object detection. In terms of model architecture, voxel-based methods can be categorized into two groups, i.e., SpCNN-based [69, 77, 53, 54, 8, 10] and Transformers-based [64, 13, 25, 41, 37] ones. Limited by the high computation complexity, SpCNN-based methods can only use small convolution kernels with restricted receptive fields, and Transformer based methods can only employ a small number of voxels in each group. In contrast, our proposed Voxel Mamba can capture long-range dependencies within the entire sequence while achieving faster inference speed than existing state-of-the-art methods.

State Space Models. Inspired by the continuous state space models (SSMs) in control systems, researchers [18, 21, 58, 20] have introduced the SSMs into deep neural networks as a novel alternative to CNNs and Transformers. LSSLs [22] adopts a simple sequence-to-sequence transformation, demonstrating the potential of SSMs. S4 [21] introduces a new parameterization method to SSMs to reduce the computation and memory cost. S5 [58] employs MIMO SSMs and perform efficient parallel scans based on S4. More recently, Mamba [20] introduces input-dependent SSMs and builds a generic backbone, which is fairly competitive with the well-tuned Transformers. Vision Mamba [80] employs bidirectional SSMs and position embedding to learn global visual context for vision tasks. Vmamba [35] employs a 2D-selective-scan to bridge the gap between 1D scanning and 2D plain traversing. PointMamba [33] is a pioneering work to leverage SSM for point cloud analysis, achieving impressive performance in point cloud object understanding. Subsequently, many SSM-based methods [24, 34, 76, 75, 67] are introduced for point cloud processing. In this paper, we investigate the utilization of SSMs to establish a straightforward yet robust baseline for LiDAR-based 3D object detection in driving scenes.

Space-filling Curve. The space-filling curve [42] is a series of fractal curves that can go through each point in a multi-dimensional space without repetition. The classical space-filling curve includes Hilbert curve [28], Z-order curve [45], and sweep curve, etc. Those methods can perform dimension reduction while maintaining spatial topology and locality. Many researchers [5, 68, 65, 37, 64, 3, 39] have introduced space-filling curves for point cloud processing. HilbertNet [5] uses the Hilbert curve to collapse 3D structures into 2D space to reduce computation and GPU occupation. PointGPT [3] utilizes the Morton-order curve [43] to introduce sequential properties. OctFormer [65] preserves Z-order during octreelization and adopts octree-attention for efficient context learning. PTV3 [68] streamlines the complex interaction with the space-filling curve serialization. For 3D object detection, some methods [64, 37] employ window sweep curves to group voxel features for parallel computation. We employ the Hilbert curve due to its advantageous characteristic of locality preservation.

Point Cloud Grouping. LiDAR point clouds are sparsely and non-uniformly distributed with varying densities. Therefore, existing methods group points or voxels to facilitate parallel computation and reduce complexity. In point cloud analysis,some works [50,66] use the $K$ nearest neighbor (KNN) method to create groups of query points. However, the heavy computation burden makes KNN hard to scale for outdoor scenes. For 3D object detection, VoTr [41] uses a GPU-based hash table to search neighborhoods and generate fixed-length voxel groups. Window-based Voxel Transformers [13, 60, 64, 37] group voxels by employing a window-based sorting strategy, such as the rotating partition. To reduce the reliance on relative position in grouping operations, some recent works [65, 68, 39] have been proposed to group voxels based on space-filling curves. However, grouping is merely a compromise for computational complexity, which restricts the flow of information and effective receptive field. To tackle this problem, we model the entire voxels into one single sequence and allow each voxel be aware of global context information.

## 3 Methods

In this section, we present Voxel Mamba, a group-free Voxel State Space Model-based 3D backbone that can be applied to most voxel-based 3D detectors. We first introduce the preliminary concepts associated with our method, followed by the overall architecture of Voxel Mamba. Then, we describe in detail the fundamental components of Voxel Mamba, including the Hilbert Input Layer (HIL), Dual-scale SSM Block (DSB), and Implicit Window Partition (IWP).

<!-- Meanless: 3 -->


<!-- Media -->

<!-- figureText: DSB 1<br>Voxel Mamba<br>DSB $N$<br>BEV Backbone<br>Detection Head<br>Output boxes<br>Input point clouds<br>Voxelization<br>Forward SSM<br>Forward SSM<br>...<br>Backward SSM<br>Down<br>Backward SSM<br>Up<br>$\times \frac{1}{{d}_{1}}$<br>$\times {d}_{1}$<br>$\times \frac{1}{{d}_{N}}$<br>$\times {d}_{N}$<br>Hilbert Input Layer<br>Non-empty Voxels ☐ Empty Vox<br>Upsampling<br>Down Sampling<br>LayerNorm<br>个<br>LayerNorm<br>A<br>SiLu<br>SiLu<br>Forward SSM<br>Backward SSM<br>A<br>个<br>个<br>7<br>Conv1d<br>Linear<br>Hilbert Input Layer<br>Linear<br>Hilbert Input Layer<br>个<br>Downsampling<br>个<br>Begin<br>End<br>Begin<br>End<br>Dual-scale SSM Block (DSB)<br>Forward SSM Input Sequence<br>Backward SSM Input Sequence<br>State Space Model (SSM) -->

<img src="https://cdn.noedgeai.com/bo_d8buj02lb0pc73eb040g_3.jpg?x=317&y=211&w=1175&h=585&r=0"/>

Figure 2: Top: The overall architecture of our proposed Voxel Mamba with $N$ Dual-scale SSM Blocks (DSBs). Bottom: Illustration of the DSB, including a residual connection, a forward SSM branch, and a backward SSM branch.

<!-- Media -->

### 3.1 Preliminaries

The state space sequence (SSM) model is a continuous-time latent state model, which maps a 1D input signal $x\left( t\right)  \in  {\mathbb{R}}^{L}$ to an output signal $y\left( t\right)  \in  {\mathbb{R}}^{L}$ through hidden state $h\left( t\right)  \in  {\mathbb{R}}^{N}$ . The system can be represented as the following linear ordinary differential equation:

$$
\begin{cases} {h}^{\prime }\left( t\right) &  = \mathbf{A}h\left( t\right)  + \mathbf{B}x\left( t\right) , \\  y\left( t\right) &  = \mathbf{C}h\left( t\right)  + \mathbf{D}x\left( t\right) , \end{cases} \tag{1}
$$

where $\mathbf{A} \in  {\mathbb{R}}^{N \times  N},\mathbf{B} \in  {\mathbb{R}}^{N \times  1}$ and $\mathbf{C} \in  {\mathbb{R}}^{1 \times  N}$ are learnable parameters,and $\mathbf{D} \in  {\mathbb{R}}^{1}$ denotes a residual connection.

To apply SSM to a discrete sequence, we can discrete the continuous-time SSM with a timescale parameter $\mathbf{\Delta }$ [21,20,35]. The zero-order hold (ZOH) transformation can be used to discrete the continuous parameters $\mathbf{A},\mathbf{B}$ as $\overline{\mathbf{A}} = \exp \left( {\mathbf{\Delta }\mathbf{A}}\right) ,\overline{\mathbf{B}} = {\left( \mathbf{\Delta }\mathbf{A}\right) }^{-\mathbf{1}}\left( {\exp \left( {\mathbf{\Delta }\mathbf{A}}\right)  - \mathbf{I}}\right)  \cdot  \mathbf{\Delta }\mathbf{A}$ . The discretized version of Eq. 1 can be written in the following recurrent form:

$$
\left\{  \begin{array}{l} {h}_{k} = \overline{\mathbf{A}}{h}_{k - 1} + \overline{\mathbf{B}}{x}_{k}, \\  {y}_{k} = \overline{\mathbf{C}}{h}_{k} + \overline{\mathbf{D}}{x}_{k}. \end{array}\right. \tag{2}
$$

Finally, the convolutional mode can be used for efficient parallel training:

$$
\left\{  \begin{array}{l} \overline{\mathbf{K}} = \left( {\mathbf{C}\overline{\mathbf{B}},\mathbf{{CA}}\overline{\mathbf{{AB}}},\ldots ,\mathbf{C}{\overline{\mathbf{A}}}^{\mathbf{L}}\overline{\mathbf{B}}}\right) , \\  \mathbf{y} = \mathbf{x} * \overline{\mathbf{K}}, \end{array}\right. \tag{3}
$$

where $\mathrm{L}$ is the length of the input sequence and $\bar{K} \in  {\mathbb{R}}^{L}$ is the structured convolution kernel.

SSM combines the advantages of convolution and self-attention with near-linear computation and dynamic weights. It demonstrates stronger ability than Transformers in modeling long-range dependencies [21, 20], which inspires us to develop a group-free framework for point cloud based 3D object detection.

<!-- Meanless: 4 -->


### 3.2 Overall Architecture

An overview of our proposed Voxel Mamba is shown in Figure 2 As in previous works [64, 73, 30], Voxel Mamba transforms point clouds into sparse voxels by a voxel feature encoding strategy. Unlike prior Transformer-based methods that perform extensive window partitioning and voxel grouping, in Voxel Mamba we serialize the voxel of the entire scene into a single sequence by using the Hilbert Input Layer (Sec. 3.3). Then, a Dual-scale SSM Block (Sec. 3.4) working on the voxel sequence is proposed, which allows voxels to be processed with a global context. To enlarge the effective receptive fields, DSB adopts a finer-grained perception of the voxel sequence in the forward path, and down-samples the voxels sequence in the backward path. The backward path extracts features from the low-resolution BEV representation, with an increased downsampling factor in deeper blocks. To enhance the spatial proximity in sequences, Voxel Mamba adopts Implicit Window Partition (Sec. 3.5) to preserve 3D positional information in the extracted voxel features, and projects them to a BEV feature map. Our proposed architecture is flexible and can be applied to most existing 3D object detection frameworks.

### 3.3 Hilbert Input Layer

The space-filling curve (e.g., Hilbert [28] and Z-order [45]), known for preserving spatial locality, is widely used for dimensionality reduction. Space-filling curves, such as the Hilbert shown in Fig. 2 can traverse all elements in a space without repetition and preserve spatial topology. To improve the voxel proximity in serialization, we propose the Hilbert Input Layer to reorder the voxel sequence.

Denote the coordinates of voxel features as $\mathcal{C} = \left\{  {\left( {x,y,z}\right)  \in  {\mathbb{R}}^{3} \mid  0 \leq  x,y,z \leq  n}\right\}$ . We map a voxel onto its traversal position $h$ within the Hilbert curve. Specifically,we transform $\left( {x,y,z}\right)$ into its binary format with ${\log }_{2}n$ bits. For example, $x$ is converted to $\left( {{x}_{m}{x}_{m - 1}\ldots {x}_{0}}\right)$ ,where $m = \left\lfloor  {{\log }_{2}n}\right\rfloor$ . Then,following [57],we iterate from ${x}_{m},{y}_{m},{z}_{m}$ to ${x}_{1},{y}_{1},{z}_{1}$ bits and perform exchanges and inversions to adjust the order of bits. An exchange is conducted when the current bit is 0 ; otherwise, an invert is conducted. We concatenate all bits as $\left( {{x}_{m}{y}_{m}{z}_{m}{x}_{m - 1}{y}_{m - 1}{z}_{m - 1}\ldots {x}_{0}{y}_{0}{z}_{0}}\right)$ and apply a global ${3m}$ -fold Gray decoding [57] on it to obtain the traversal position $h$ . Subsequently,all voxels are sorted into a single sequence based on their traversal position $h$ .

In our implementation,we record the traversal position $h$ corresponding to the coordinates of all potential voxels. The voxels are serialized by querying and sorting their traversal positions. Notably, the serialization process only takes approximately ${0.7}\mathrm{\;{ms}}$ for a sequence of length ${10}^{6}$ .

### 3.4 Dual-scale SSM Block

Though space-filling curves can preserve the 3D structure to a certain degree, proximity loss is inevitable due to the dimension collapse from 3D to 1D. As a result, a local snippet of the curve can only cover a partial region in 3D space. As discussed in Sec. 1, placing all voxels in a single group cannot ensure that the effective receptive field (ERF) [40, 12] could cover all voxels. Therefore, in this subsection we introduce the Dual-scale SSM block (DSB) to build a hierarchy of state space structures and consequently improve the ERF of the model.

As shown in Fig. 2, the DSB block is designed with a residual connection [27], a forward SSM branch and a backward SSM branch. It operates on two serialized voxel sequences generated by the Hilbert Input Layer, enabling a seamless flow of information throughout the voxel sequence. The forward branch processes the original voxel sequence, maintaining high-resolution details. The backward branch, however, operates on a down-sampled voxel sequence derived from a low-resolution BEV representation. This dual-scale path allows DSB to incorporate larger-scale voxel features, enhancing the model's ability to model long dependencies among voxels. Specifically, given a voxel sequence $\mathcal{F}$ and its corresponding coordinates $\mathcal{C}$ ,DSB is computed as:

$$
{\mathcal{F}}_{f} = \mathbf{{LN}}\left( {\operatorname{FSSM}\left( {\operatorname{HIL}\left( {\mathcal{F} + \operatorname{IWE}\left( \mathcal{C}\right) }\right) }\right) }\right)
$$

$$
{\mathcal{F}}_{b} = \mathbf{{Up}}\left( {\mathbf{{LN}}\left( {\operatorname{BSSM}\left( {\operatorname{HIL}\left( {\operatorname{Down}\left( \mathcal{F}\right)  + \mathbf{{IWE}}\left( {\mathcal{C}}^{\prime }\right) }\right) }\right) }\right) }\right) , \tag{4}
$$

$$
\widetilde{\mathcal{F}} = {\mathcal{F}}_{f} + {\mathcal{F}}_{b} + \mathcal{F},
$$

where $\operatorname{HIL}\left( \cdot \right)$ represents the Hilbert Input Layer, $\operatorname{\mathbf{F} \mathbf{S} \mathbf{S} \mathbf{M} }\left( \cdot \right)$ and $\operatorname{\mathbf{B} \mathbf{S} \mathbf{S} \mathbf{M} }\left( \cdot \right)$ denote the forward and backward SSM,LN $\left( \cdot \right)$ stands for Layer Normalization,and ${\mathcal{C}}^{\prime }$ is the coordinates of downsampled sparse voxels. Besides, Down $\left( \cdot \right)$ and $\mathbf{{Up}}\left( \cdot \right)$ refer to the downsampling and upsampling operations, respectively, and IWE(·) means Implicit Window Embedding. Overall, DSB integrates the widely adopted bidirectional design [80, 35] with the hierarchical design, building sufficient receptive field to mitigate the loss of proximity without introducing additional parameters.

<!-- Meanless: 5 -->


### 3.5 Implicit Window Partition

The window partition strategy is widely used in previous 3D detectors [13, 64] to enhance the voxel proximity. In these methods, the whole field is partitioned into multiple local windows and the voxels within a window form a group. Therefore, the voxels inside a window will have sufficient proximity; however, the voxels in different windows will have minimal proximity. In this section, we aim to introduce the advantages of window partition into our framework while avoiding its weaknesses.

To fulfill our goal, we propose an Implicit Window Partition (IWP) strategy. Unlike previous methods, we do not explicitly partition voxels into windows and apply Transformer or SSM within each window. In contrast, we calculate the voxel coordinates inside and across windows, and then encode coordinates to embeddings, termed as Implicit Window Embedding (IWE), which is formulated as:

$$
\mathbf{{IWE}} = \mathbf{{MLP}}\left( {\operatorname{concat}\left( {z,\left\lfloor  \frac{{x}^{i}}{w}\right\rfloor  ,\left\lfloor  \frac{{y}^{i}}{h}\right\rfloor  ,{x}^{i}{\;\operatorname{mod}\;w},{y}^{i}{\;\operatorname{mod}\;h}}\right) }\right) ,i = 0,1 \tag{5}
$$

where $\lfloor  \cdot  \rfloor$ is the floor function, $w,h$ define the window shape,and $z,{x}^{i},{y}^{i}$ are the coordinates of tokens. $\left( {{x}^{0},{y}^{0}}\right)$ and $\left( {{x}^{1},{y}^{1}}\right)$ represent the coordinates before and after an implicit window shift. The IWE is shared across all layers with the same stride. Thus, its computation cost only comes from shallow MLPs. With IWE, voxels in the serialized 1D curve are aware of their positions and consequently their proximity in 3D space.

### 3.6 The Voxel Mamba Backbone

With the proposed Hilbert Input Layer, DSB and IWP strategies, we build Voxel Mamba, a group-free sparse voxel backbone. The architecture of Voxel Mamba is illustrated in Figure 2 It comprises $N$ DSB blocks, which are organized into different stages based on their downsampling rates. SpConv [9] is employed to progressively decrease the feature map resolution along the Z-axis in each stage. Before sparse tokens are fed into the BEV backbone, we scatter them into dense BEV features. On the Waymo dataset, we adopt the BEV backbone from Centerpoint-Pillar [73], and employ the same setting as DSVT [64] for the detection head and loss function. On the nuScenes dataset, we only replace the 3D backbone of DSVT [64] with our Voxel Mamba backbone.

## 4 Experiments

### 4.1 Datasets and Evaluation Metrics

Waymo Open Dataset contains ${230}\mathrm{k}$ annotated samples,partitioned into ${160}\mathrm{k}$ for training, ${40}\mathrm{k}$ for validation and ${30}\mathrm{k}$ for testing. Each frame covers a large perception range $\left( {{150}\mathrm{\;m} \times  {150}\mathrm{\;m}}\right)$ . The mean average precision (mAP) and its weighted variant by heading accuracy (mAPH) are used as evaluation metrics. They are further categorized into Level 1 for objects detected by over five points, and Level 2 for those detected with at least one point.

nuScenes consists of 40k labeled samples,with ${28}\mathrm{k}$ for training, $6\mathrm{k}$ for validation and $6\mathrm{k}$ for testing. For 3D object detection, nuScenes employs the mean average precision (mAP) and the nuScenes detection score (NDS) to measure model performance.

### 4.2 Implementation Details

Our method is implemented based on the open-source framework OpenPCDet [62]. The voxel sizes are defined as $\left( {{0.32m},{0.32m},{0.1875m}}\right)$ for Waymo and $\left( {{0.3m},{0.3m},{0.2m}}\right)$ for nuScenes. We stack six DSB blocks, divided into three stages, for the Voxel Mamba backbone network. The downsampling rates for DSBs' backward branches in each stage are $\{ 1,2,4\}$ . Specifically,we employ SpConv [9] and its counterpart SpInverseConv as downsampling and upsampling operators in the DSB backward branch. On the Waymo dataset, Voxel Mamba is trained for 24 epochs with a learning rate of 0.0025 . On the nuScenes dataset, we train Voxel Mamba for 20 epochs with a learning rate of 0.004 . The voxel features on both datasets consist of 128 channels. All the models are trained by the AdamW optimizer on 8 RTX A6000 GPUs. Other settings in the training and inference of Voxel Mamba follow DSVT [64].

<!-- Meanless: 6 -->


<!-- Media -->

Table 1: Performance comparison on the validation set of Waymo Open Dataset (single-frame setting). Symbol '-' means that the result is not available.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Category</td><td colspan="2">ALL (3D mAPH)</td><td colspan="2">Vehicle (AP/APH)</td><td colspan="2">Pedestrian (AP/APH)</td><td colspan="2">Cyclist (AP/APH)</td></tr><tr><td>L1</td><td>L2</td><td>L1</td><td>L2</td><td>L1</td><td>L2</td><td>L1</td><td>L2</td></tr><tr><td>PointPillar 30</td><td rowspan="3">2D CNN</td><td>63.3</td><td>57.5</td><td>71.6 / 71.0</td><td>63.1 / 62.5</td><td>70.6 / 56.7</td><td>62.9 / 50.2</td><td>64.4 / 62.3</td><td>61.9 / 59.9</td></tr><tr><td>Centerpoint-Pillar 73</td><td>-</td><td>-</td><td>76.1 / 75.5</td><td>68.0 / 67.5</td><td>-</td><td>- / 62.6</td><td>-</td><td>- / 67.6</td></tr><tr><td>PillarNeXt [31]</td><td>75.7</td><td>69.7</td><td>78.4 / 77.9</td><td>70.3 / 69.8</td><td>82.5 / 77.1</td><td>74.9 / 69.8</td><td>73.2 / 72.2</td><td>70.6 / 69.6</td></tr><tr><td>SECOND 69</td><td rowspan="11">SpCNN</td><td>63.1</td><td>57.2</td><td>72.3 / 71.7</td><td>63.9 / 63.3</td><td>68.7 / 58.2</td><td>60.7 / 51.3</td><td>60.6 / 59.3</td><td>58.3 / 57.1</td></tr><tr><td>Part-A2 56</td><td>70.3</td><td>63.8</td><td>77.1 / 76.5</td><td>68.5 / 68.0</td><td>75.2 / 66.9</td><td>66.2 / 58.6</td><td>68.6 / 67.4</td><td>66.1 / 64.9</td></tr><tr><td>PV-RCNN [53]</td><td>69.6</td><td>63.3</td><td>77.5 / 76.9</td><td>69.0 / 68.4</td><td>75.0 / 65.7</td><td>66.0 / 57.6</td><td>67.8 / 66.4</td><td>65.4 / 64.0</td></tr><tr><td>Centerpoint-Voxel [73]</td><td>-</td><td>67.6</td><td>76.6 / 76.0</td><td>68.9 / 68.4</td><td>79.0 / 73.4</td><td>71.0 / 65.8</td><td>72.1 / 71.0</td><td>69.5 / 68.5</td></tr><tr><td>PV-RCNN++ [54]</td><td>75.2</td><td>68.6</td><td>79.1 / 78.6</td><td>70.3 / 69.9</td><td>80.6 / 74.6</td><td>71.9 / 66.3</td><td>73.5 / 72.4</td><td>70.7 / 69.6</td></tr><tr><td>AFDetV2 29</td><td>74.8</td><td>68.8</td><td>77.6 / 77.1</td><td>69.7 / 69.2</td><td>80.2 / 74.6</td><td>72.2 / 67.0</td><td>73.7 / 72.7</td><td>71.0 / 70.1</td></tr><tr><td>VoxelNeXt 8</td><td>76.3</td><td>70.1</td><td>78.2 / 77.7</td><td>69.9 / 69.4</td><td>81.5 / 76.3</td><td>73.5 / 68.6</td><td>76.1 / 74.9</td><td>73.3 / 72.2</td></tr><tr><td>HEDNet 74</td><td>79.4</td><td>73.4</td><td>81.1 / 80.6</td><td>73.2 / 72.7</td><td>84.4 / 80.0</td><td>76.8 / 72.6</td><td>78.7 / 77.7</td><td>75.8 / 74.9</td></tr><tr><td>PillarNet 52</td><td>74.6</td><td>68.4</td><td>79.1 / 78.6</td><td>70.9 / 70.5</td><td>80.6 / 74.0</td><td>72.3 / 66.2</td><td>72.3 / 71.2</td><td>69.7 / 68.7</td></tr><tr><td>FSD 14</td><td>77.3</td><td>70.8</td><td>79.2 / 78.8</td><td>70.5 / 70.1</td><td>82.6 / 77.3</td><td>73.9 / 69.1</td><td>77.1 / 76.0</td><td>74.4 / 73.3</td></tr><tr><td>ConQueR 79</td><td>77.9</td><td>71.6</td><td>78.4 / 77.9</td><td>71.0 / 70.5</td><td>82.4 / 76.6</td><td>75.8 / 70.1</td><td>77.5 / 76.4</td><td>75.2 / 74.1</td></tr><tr><td>VoTR 41</td><td rowspan="8">Group-based</td><td>-</td><td>-</td><td>75.0 / 74.3</td><td>65.9 / 65.3</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VoxSeT 25</td><td>72.2</td><td>66.2</td><td>74.5 / 74.0</td><td>66.0 / 65.6</td><td>80.0 / 72.4</td><td>72.5 / 65.4</td><td>71.6 / 70.3</td><td>69.0 / 67.7</td></tr><tr><td>SST 13</td><td>-</td><td>-</td><td>76.2 / 75.8</td><td>68.0 / 67.6</td><td>81.4 / 74.1</td><td>72.8 / 65.9</td><td>-</td><td>-</td></tr><tr><td>SWFormer 60</td><td>-</td><td>-</td><td>77.8 / 77.3</td><td>69.2 / 68.8</td><td>80.9 / 72.7</td><td>72.5 / 64.9</td><td>-</td><td>-</td></tr><tr><td>CenterFormer | 78</td><td>73.2</td><td>69.1</td><td>75.0 / 74.4</td><td>69.9 / 69.4</td><td>78.0 / 72.4</td><td>73.1 / 67.7</td><td>73.8 / 72.7</td><td>71.3 / 70.2</td></tr><tr><td>FlatFormer 37</td><td>-</td><td>67.2</td><td>-</td><td>69.0 / 68.6</td><td>-</td><td>71.5 / 65.3</td><td>-</td><td>68.6 / 67.5</td></tr><tr><td>PTv3 68</td><td>-</td><td>70.5</td><td>-</td><td>71.2 / 70.8</td><td>-</td><td>76.3 / 70.4</td><td>-</td><td>71.5 / 70.4</td></tr><tr><td>DSVT-Voxel 64</td><td>78.2</td><td>72.1</td><td>79.7 / 79.3</td><td>71.4 / 71.0</td><td>83.7 / 78.9</td><td>76.1 / 71.5</td><td>77.5176.5</td><td>74.6 / 73.7</td></tr><tr><td>Voxel Mamba (ours)</td><td>Group-free</td><td>79.6</td><td>73.6</td><td>80.8 / 80.3</td><td>72.6 / 72.2</td><td>85.0 / 80.8</td><td>77.7 / 73.6</td><td>78.6 / 77.6</td><td>75.7 / 74.8</td></tr></table>

Table 2: Performance comparison on the test set of Waymo Open Dataset. Symbol '-' means that the result is not available. "3f" stands for 3-frame model.

<table><tr><td rowspan="2">Method</td><td colspan="2">ALL (3D mAPH)</td><td colspan="2">Vehicle (AP / APH)</td><td colspan="2">Pedestrian (AP / APH)</td><td colspan="2">Cyclist (AP / APH)</td></tr><tr><td>L1</td><td>L2</td><td>L1</td><td>L2</td><td>L1</td><td>L2</td><td>L1</td><td>L2</td></tr><tr><td>PointPillar 30</td><td>-</td><td>-</td><td>68.6 / 68.1</td><td>60.5 / 60.1</td><td>68.0 / 55.5</td><td>61.4 / 50.1</td><td>-</td><td>-</td></tr><tr><td>M3DETR 23</td><td>67.1</td><td>61.9</td><td>77.7 / 77.1</td><td>70.5 / 70.0</td><td>68.2 / 58.5</td><td>60.6 / 52.0</td><td>67.3 / 65.7</td><td>65.3 / 63.8</td></tr><tr><td>3D-MAN 72</td><td>-</td><td>-</td><td>78.7 / 78.3</td><td>70.4 / 70.0</td><td>70.0 / 66.0</td><td>64.0 / 60.3</td><td>-</td><td>-</td></tr><tr><td>PV-RCNN++ 54</td><td>75.7</td><td>70.2</td><td>81.6 / 81.2</td><td>73.9 / 73.5</td><td>80.4 / 75.0</td><td>74.1 / 69.0</td><td>71.9 / 70.8</td><td>69.3 / 68.2</td></tr><tr><td>CenterPoint 73</td><td>77.2</td><td>71.9</td><td>81.1 / 80.6</td><td>73.4 / 73.0</td><td>80.5 / 77.3</td><td>74.6 / 71.5</td><td>74.6 / 73.7</td><td>72.2 / 71.3</td></tr><tr><td>RSN [61]</td><td>-</td><td>-</td><td>80.7 / 80.3</td><td>71.9 / 71.6</td><td>78.9 / 75.6</td><td>70.7 / 67.8</td><td>-</td><td>-</td></tr><tr><td>SST-3f [13]</td><td>78.3</td><td>72.8</td><td>81.0 / 80.6</td><td>73.1 / 72.7</td><td>83.3 / 79.7</td><td>76.9 / 73.5</td><td>75.7 / 74.6</td><td>73.2 / 72.2</td></tr><tr><td>Graph-RCNN [70]</td><td>77.0</td><td>71.6</td><td>83.6 / 83.1</td><td>76.0 / 75.6</td><td>81.9 / 76.5</td><td>75.6 / 70.5</td><td>72.5 / 71.3</td><td>69.8 / 68.7</td></tr><tr><td>FSDv1 14</td><td>78.2</td><td>72.4</td><td>82.7 / 82.3</td><td>74.4 / 74.1</td><td>82.9 / 77.9</td><td>75.9 / 71.3</td><td>75.6 / 74.4</td><td>72.9 / 71.8</td></tr><tr><td>FSDv2 15</td><td>79.0</td><td>73.3</td><td>82.4 / 82.0</td><td>74.4 / 74.0</td><td>83.8 / 78.9</td><td>77.4 / 72.8</td><td>77.1 / 76.0</td><td>74.3 / 73.2</td></tr><tr><td>PillarNeXt-3f 31</td><td>79.0</td><td>74.1</td><td>83.3 / 82.8</td><td>76.2 / 75.8</td><td>84.4 / 81.4</td><td>78.8 / 76.0</td><td>73.8 / 72.7</td><td>71.6 / 70.6</td></tr><tr><td>Voxel Mamba (ours)</td><td>79.6</td><td>74.3</td><td>84.4 / 84.0</td><td>77.0 / 76.6</td><td>84.8 / 80.6</td><td>79.0 / 74.9</td><td>75.4 / 74.3</td><td>72.6 / 71.5</td></tr></table>

<!-- Media -->

### 4.3 Comparison with State-of-the-art Methods

Waymo. We first compare Voxel Mamba with state-of-the-art methods on the Waymo Open dataset. Table 1 shows the results on the validation set. Our proposed Voxel Mamba achieves 79.6/73.4 on L1TL2 mAPH, which are +1.4 and +1.5 better than DSVT-Voxel. Since our framework differs from DSVT only on the 3D backbone, it can be concluded that Voxel Mamba has superior ability in capturing voxel features. In comparison with window-based (e.g., DSVT) or curve-based (e.g., PTv3) grouping methods, our group-free method Voxel Mamba consistently delivers better results. Table 2 shows the results on the test split. Voxel Mamba reaches 79.6/74.3 in terms of L1/L2 mAPH, which is even better than the 3-frame setting of PillarNeXt and SST.

nuScenes. We then compare Voxel Mamba with previous state-of-the-art methods on nuScenes. Table 3 shows the results on the validation set. Voxel Mamba achieves impressive results with 71.9 NDS and 67.5 mAP, which is +0.5 and +0.8 higher than the previous best method. Compared with DSVT, Voxel Mamba achieves +1.1 higher performance on mAP. The results on the test split are shown in Table 4 Our method also exhibits the best mAP and NDS.

<!-- Meanless: 7 -->


<!-- Media -->

Table 3: Comparison with the state-of-the-art detectors on the nuScenes dataset validation split.

<table><tr><td>Method</td><td>NDS</td><td>mAP</td><td>Car</td><td>Truck</td><td>Bus</td><td>T.L.</td><td>C.V.</td><td>Ped.</td><td>M.T.</td><td>Bike</td><td>T.C.</td><td>B.R.</td></tr><tr><td>CenterPoint [73]</td><td>66.5</td><td>59.2</td><td>84.9</td><td>57.4</td><td>70.7</td><td>38.1</td><td>16.9</td><td>85.1</td><td>59.0</td><td>42.0</td><td>69.8</td><td>68.3</td></tr><tr><td>VoxelNeXt [8]</td><td>66.7</td><td>60.5</td><td>83.9</td><td>55.5</td><td>70.5</td><td>38.1</td><td>21.1</td><td>84.6</td><td>62.8</td><td>50.0</td><td>69.4</td><td>69.4</td></tr><tr><td>TransFusion-L [1]</td><td>70.1</td><td>65.5</td><td>86.9</td><td>60.8</td><td>73.1</td><td>43.4</td><td>25.2</td><td>87.5</td><td>72.9</td><td>57.3</td><td>77.2</td><td>70.3</td></tr><tr><td>PillarNeXt [31]</td><td>68.4</td><td>62.2</td><td>85.0</td><td>57.4</td><td>67.6</td><td>35.6</td><td>20.6</td><td>86.8</td><td>68.6</td><td>53.1</td><td>77.3</td><td>69.7</td></tr><tr><td>HEDNet [74]</td><td>71.4</td><td>66.7</td><td>87.7</td><td>60.6</td><td>77.8</td><td>50.7</td><td>28.9</td><td>87.1</td><td>74.3</td><td>56.8</td><td>76.3</td><td>66.9</td></tr><tr><td>DSVT 64</td><td>71.1</td><td>66.4</td><td>87.4</td><td>62.6</td><td>75.9</td><td>42.1</td><td>25.3</td><td>88.2</td><td>74.8</td><td>58.7</td><td>77.8</td><td>70.9</td></tr><tr><td>Voxel Mamba (ours)</td><td>71.9</td><td>67.5</td><td>87.9</td><td>62.8</td><td>76.8</td><td>45.9</td><td>24.9</td><td>89.3</td><td>77.1</td><td>58.6</td><td>80.1</td><td>71.5</td></tr></table>

Table 4: Comparison with the state-of-the-art detectors on the nuScenes dataset test split.

<table><tr><td>Method</td><td>NDS</td><td>mAP</td><td>Car</td><td>Truck</td><td>Bus</td><td>T.L.</td><td>C.V.</td><td>Ped.</td><td>M.T.</td><td>Bike</td><td>T.C.</td><td>B.R.</td></tr><tr><td>PointPillars 30</td><td>45.3</td><td>30.5</td><td>68.4</td><td>23.0</td><td>28.2</td><td>23.4</td><td>4.1</td><td>59.7</td><td>27.4</td><td>1.1</td><td>30.8</td><td>38.9</td></tr><tr><td>3DSSD [71]</td><td>56.4</td><td>42.6</td><td>81.2</td><td>47.2</td><td>61.4</td><td>30.5</td><td>12.6</td><td>70.2</td><td>36.0</td><td>8.6</td><td>31.1</td><td>47.9</td></tr><tr><td>CenterPoint [73]</td><td>65.5</td><td>58.0</td><td>84.6</td><td>51.0</td><td>60.2</td><td>53.2</td><td>17.5</td><td>83.4</td><td>53.7</td><td>28.7</td><td>76.7</td><td>70.9</td></tr><tr><td>FCOS-LiDAR [63]</td><td>65.7</td><td>60.2</td><td>82.2</td><td>47.7</td><td>52.9</td><td>48.8</td><td>28.8</td><td>84.5</td><td>68.0</td><td>39.0</td><td>79.2</td><td>70.7</td></tr><tr><td>AFDetV2 29</td><td>68.5</td><td>62.4</td><td>86.3</td><td>54.2</td><td>62.5</td><td>58.9</td><td>26.7</td><td>85.8</td><td>63.8</td><td>34.3</td><td>80.1</td><td>71.0</td></tr><tr><td>UVTR-L 32</td><td>69.7</td><td>63.9</td><td>86.3</td><td>52.2</td><td>62.8</td><td>59.7</td><td>33.7</td><td>84.5</td><td>68.8</td><td>41.1</td><td>74.7</td><td>74.9</td></tr><tr><td>VISTA [11]</td><td>69.8</td><td>63.0</td><td>84.4</td><td>55.1</td><td>63.7</td><td>54.2</td><td>25.1</td><td>82.8</td><td>70.0</td><td>45.4</td><td>78.5</td><td>71.4</td></tr><tr><td>Focals Conv [6]</td><td>70.0</td><td>63.8</td><td>86.7</td><td>56.3</td><td>67.7</td><td>59.5</td><td>23.8</td><td>87.5</td><td>64.5</td><td>36.3</td><td>81.4</td><td>74.1</td></tr><tr><td>VoxelNeXt [8</td><td>70.0</td><td>64.5</td><td>84.6</td><td>53.0</td><td>64.7</td><td>55.8</td><td>28.7</td><td>85.8</td><td>73.2</td><td>45.7</td><td>79.0</td><td>74.6</td></tr><tr><td>TransFusion-L 1</td><td>70.2</td><td>65.5</td><td>86.2</td><td>56.7</td><td>66.3</td><td>58.8</td><td>28.2</td><td>86.1</td><td>68.3</td><td>44.2</td><td>82.0</td><td>78.2</td></tr><tr><td>LinK 38</td><td>71.0</td><td>66.3</td><td>86.1</td><td>55.7</td><td>65.7</td><td>62.1</td><td>30.9</td><td>85.8</td><td>73.5</td><td>47.5</td><td>80.4</td><td>75.5</td></tr><tr><td>HEDNet [74]</td><td>72.0</td><td>67.7</td><td>87.1</td><td>56.5</td><td>70.4</td><td>63.5</td><td>33.6</td><td>87.9</td><td>70.4</td><td>44.8</td><td>85.1</td><td>78.1</td></tr><tr><td>LargeKernel3D [7]</td><td>70.6</td><td>65.4</td><td>85.5</td><td>53.8</td><td>64.4</td><td>59.5</td><td>29.7</td><td>85.9</td><td>72.7</td><td>46.8</td><td>79.9</td><td>75.5</td></tr><tr><td>PillarNet 52</td><td>71.4</td><td>66.0</td><td>87.6</td><td>57.5</td><td>63.6</td><td>63.1</td><td>27.9</td><td>87.3</td><td>70.1</td><td>42.3</td><td>83.3</td><td>77.2</td></tr><tr><td>FSDv2 [15]</td><td>71.7</td><td>66.2</td><td>83.7</td><td>51.6</td><td>66.4</td><td>59.1</td><td>32.5</td><td>87.1</td><td>71.4</td><td>51.7</td><td>80.3</td><td>78.7</td></tr><tr><td>DSVT [64]</td><td>72.7</td><td>68.4</td><td>86.8</td><td>58.4</td><td>67.3</td><td>63.1</td><td>37.1</td><td>88.0</td><td>73.0</td><td>47.2</td><td>84.9</td><td>78.4</td></tr><tr><td>Voxel Mamba (ours)</td><td>73.0</td><td>69.0</td><td>86.8</td><td>57.1</td><td>68.0</td><td>63.2</td><td>35.4</td><td>89.5</td><td>74.7</td><td>50.8</td><td>86.9</td><td>77.3</td></tr></table>

<!-- Media -->

Inference Efficiency. We compare Voxel Mamba with other state-of-the-art methods in inference speed and performance accuracy in Fig. 3 Notably, Voxel Mamba outperforms DSVT [64] and PV-RCNN++ [53] by at least +1.5 in detection accuracy, while achieving faster speed. Some methods, such as CenterPoint [73] and PointPillar [30], are faster than Voxel Mamba; however, their accuracy is substantially lower.

We further compare Voxel Mamba with previous well-designed architectures (SpCNN, Transformers, and 2D CNN) in GPU memory in Table 5 Compared with CenterPoint-Pillar, Voxel Mamba requires only an additional 0.5 GB GPU memory but achieves +9.0 higher accuracy in L2 mAPH. While Transformer-based methods like SST [13] and DSVT [64] use group partitioning, they still consume more memory than our group-free Voxel Mamba. All the experiments are evaluated on an NVIDIA A100 GPU with the same environment.

### 4.4 Ablation Studies

To better investigate the effectiveness of Voxel Mamba, we conduct a set of ablation studies by using the nuScenes validation set. We follow OpenPCDet [62] to train all models for 20 epochs.

Effectiveness of Space-filling Curves. There are some potential alternatives to Hilbert curve for preserving locality. Here, we compare Hilbert curve with some commonly used space-filling curves (Z-order [65] and window partition [13]) in 3D detection. As shown in Table 6(a), without using space-filling curves (i.e., the row of 'Random Curve'), there will be a notable decline in performance, which indicates that spatial proximity is crucial in the group-free setting. By using the Z-order curve and window partition to introduce spatial proximity, the mAP and NDS are much improved. The serialization based on the Hilbert curve can further enhance the model performance.

Effectiveness of Each Component. To more clearly illustrate the effectiveness of the different components in Voxel Mamba, we conduct experiments by adding each of them to a baseline, which is set to Centerpoint-Pillar [73]. As shown in Table 6(b), bidirectional SSMs with a Hilbert-based group-

<!-- Meanless: 8 -->


<!-- Media -->

<!-- figureText: Performance vs. Speed<br>Voxel Mamba (Ours)<br>72.5<br>DSVT-Voxel<br>Performance (mAP@L2)<br>70.0<br>LargeKernel3D<br>67.5<br>65.0<br>VoxSet<br>CenterPoint-Voxel<br>Part A2<br>SST<br>62.5<br>PV-RCNN<br>CenterPoint-Pillar<br>60.0<br>57.5<br>Second PointPillar<br>5<br>10<br>15<br>20<br>25<br>30<br>Frame per second (FPS) -->

<img src="https://cdn.noedgeai.com/bo_d8buj02lb0pc73eb040g_8.jpg?x=323&y=208&w=564&h=446&r=0"/>

Figure 3: Detection performance (mAPH/L2) vs. speed (FPS) on Waymo.

<table><tr><td>Abalation</td><td>Backbone</td><td>Memory (GB)</td></tr><tr><td>PointPillar 30 Centerpoint-Pillar [73]</td><td>2D CNN</td><td>3.6 <br> 3.2</td></tr><tr><td>Part-A2 [56] PV-RCNN++ (ResNet) 54</td><td>SpCNN</td><td>2.9 <br> 17.2</td></tr><tr><td>SST 13 <br> DSVT-Voxel [64]</td><td>Transformers</td><td>6.8 <br> 4.2</td></tr><tr><td>Voxel Mamba (Ours)</td><td>SSMs</td><td>3.7</td></tr></table>

<!-- Media -->

Table 5: Comparison with other well-designed architectures on GPU memory.

Table 6: Ablations on the nuScenes validation split. In (d), Centerpoint-Pillar is used as the baseline. free sequence can significantly improve the accuracy over the baseline, which validates the feasibility of our group-free strategy. Besides, converting pillar to voxel can enhance much the detector's performance without group size constraints. Voxel Mamba with DSB obtain better performance than the plain bidirectional SSMs. This is because DSB can build larger ERFs and mitigate the loss of proximity. Furthermore, IWE further boosts Voxel Mamba's performance for its capability in capturing 3D position information and increasing voxel proximity.

<!-- Media -->

<table><tr><td>Space-filing Curve</td><td>mAP</td><td>NDS</td></tr><tr><td>Random Curve</td><td>66.0</td><td>71.0</td></tr><tr><td>Window Partition</td><td>67.3</td><td>71.7</td></tr><tr><td>Z-Order Curve</td><td>67.0</td><td>71.7</td></tr><tr><td>Hilbert Curve</td><td>67.5</td><td>71.9</td></tr></table>

(a) Ablation on space-filling curves.

<table><tr><td></td><td>Downstrides mAP NDS</td></tr><tr><td>\{1,1,1\}</td><td>66.6 71.4</td></tr><tr><td>\{1,2,2\}</td><td>66.9 71.8</td></tr><tr><td>\{2,2,2\}</td><td>65.6 70.8</td></tr><tr><td>\{4,4,4\}</td><td>66.2 71.2</td></tr><tr><td>\{1,2,4\}</td><td>67.5 71.9</td></tr></table>

(c) Ablation on the downsampling rates of DSB.

<table><tr><td>Ablation</td><td>mAP</td><td>NDS</td></tr><tr><td>Baseline</td><td>63.3</td><td>69.1</td></tr><tr><td>+ bidirectional SSMs <br> (Hilbert curve)</td><td>65.8</td><td>70.9</td></tr><tr><td>+ Voxel</td><td>66.3</td><td>71.0</td></tr><tr><td>+ DSB</td><td>66.7</td><td>71.3</td></tr><tr><td>+ IWP</td><td>67.5</td><td>71.9</td></tr></table>

(b) Effect of each component in Voxel Mamba.

<table><tr><td>Pos Embeding</td><td>mAP</td><td>NDS</td></tr><tr><td>Baseline</td><td>66.7</td><td>71.3</td></tr><tr><td>Absolute position</td><td>66.9</td><td>71.2</td></tr><tr><td>Cos, Sin</td><td>66.6</td><td>71.4</td></tr><tr><td>Ours (w/o shift)</td><td>67.3</td><td>71.9</td></tr><tr><td>Ours</td><td>67.5</td><td>71.9</td></tr></table>

(d) Ablation on IWE.

<!-- Media -->

Downsampling Rates of DSB. We evaluate the impact of different downsampling rates in DSB by adjusting the stride $\left\{  {{d}_{1},{d}_{2},{d}_{3}}\right\}$ in the backward SSM branch at each stage. ${d}_{i} = 1$ means the original resolution is used. The results are shown in Table 6(c). We see that transitioning from $\{ 1,1,1\}$ to $\{ 1,2,2\}$ and to $\{ 1,2,4\}$ enhances performance due to an enlarged effective receptive field and improved proximity by using larger downsampling rates at late stages. However, DSBs with $\{ 2,2,2\}$ or $\{ 4,4,4\}$ compromise performance compared to $\{ 1,1,1\}$ ,indicating that using larger downsampling rates at early stages will lose some fine details. Thus, we set the stride as \{1,2,4\} to strike a balance between effective receptive fields and detail preservation.

Effectiveness of IWE. Table 6(d) validates the capability of IWE to enhance spatial proximity. We compare IWE with some commonly used positional embedding methods [13, 64] in 3D detection. Absolute position denotes the direct encoding of voxel coordinates using an MLP. The results demonstrate that IWE can significantly improve the detection performance by offering features with rich 3D positional and proximate information.

<!-- Meanless: 9 -->


<!-- Media -->

<!-- figureText: 1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0 -->

<img src="https://cdn.noedgeai.com/bo_d8buj02lb0pc73eb040g_9.jpg?x=318&y=204&w=1173&h=382&r=0"/>

Figure 4: The effective receptive fields (ERFs) of Voxel Mamba (left), group-based bidirectional Mamba (middle) and DSVT (right).

<!-- Media -->

### 4.5 Effective Receptive Field of Voxel Mamba

Fig. 4 illustrates the Effective Receptive Fields [40, 12] (ERFs) of window partition-based method DSVT [64], group-based bidirectional Mamba and our proposed group-free method Voxel Mamba. For clear visualization, all models take pillars as inputs. The group partition in the group-based bidirectional Mamba is configured identically to DSVT. Then, we randomly select voxels of interest from the ground truth bounding box and calculate the ERF at each non-empty voxel position. Subsequently, we merge the ERFs into a single image by taking the maximum value at each voxel location. A wider activation area indicates a larger ERF. From Fig. 4, we see that Voxel Mamba exhibits a notably larger ERF than DSVT and group-based bidirectional Mamba, which can be attributed to the benefits of group-free operation. The larger ERF can cover a more complete local region and enhance the spatial proximity in 1D sequences.

## 5 Conclusion

In this paper, we proposed Voxel Mamba, a group-free SSM-based 3D backbone for point cloud based 3D detection. We first analyzed the proximity loss of group partition in current serialization-based 3D detection methods. By taking the advantage of linear complexity of SSMs, we proposed a group-free strategy to alleviate the loss of spatial proximity in 3D to 1D serialization. We further proposed the DSB block and IWP strategy to build larger effective receptive fields and improve the spatial proximity of our Voxel Mamba framework. Experiments demonstrated that Voxel Mamba achieved state-of-the-art results on Waymo and nuScene datasets. Without elaborated optimization, our model consumed less memory than group-based Voxel Transformer methods, and our group-free strategy was more efficient and deployment-friendly than group partition. Voxel Mamba provided an efficient group-free solution for sparse point clouds for 3D tasks.

Limitations. While the proposed Voxel Mamba achieves state-of-the-art performance in point cloud based 3D object detection, it still has some limitations to be further addressed. First, in the Hilbert Input Layer, the curve templates occupy a certain amount of GPU memory, which becomes more substantial as the voxel resolution increases. Besides, a more elaborately designed downsampling and upsampling operation could improve more the model efficiency. We will investigate these problems in future work.

## References

[1] Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, and Chiew-Lan Tai. Transfusion: Robust lidar-camera fusion for 3d object detection with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1090-1099, 2022.

[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621-11631, 2020.

<!-- Meanless: 10 -->


[3] Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, and Yufeng Yue. Pointgpt: Auto-regressively generative pre-training from point clouds. Advances in Neural Information Processing Systems, 36, 2024.

[4] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. arXiv preprint arXiv:2306.16927, 2023.

[5] Wanli Chen, Xinge Zhu, Guojin Chen, and Bei Yu. Efficient point cloud analysis using hilbert curve. In European Conference on Computer Vision, pages 730-747. Springer, 2022.

[6] Yukang Chen, Yanwei Li, Xiangyu Zhang, Jian Sun, and Jiaya Jia. Focal sparse convolutional networks for 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5428-5437, 2022.

[7] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Largekernel34: Scaling up kernels in 3d sparse cnns. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13488-13498, 2023.

[8] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21674-21683, 2023.

[9] Spconv Contributors. Spconv: Spatially sparse convolution library. https://github.com/ traveller59/spconv, 2022.

[10] Jiajun Deng, Shaoshuai Shi, Peiwei Li, Wengang Zhou, Yanyong Zhang, and Houqiang Li. Voxel r-cnn: Towards high performance voxel-based 3d object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1201-1209, 2021.

[11] Shengheng Deng, Zhihao Liang, Lin Sun, and Kui Jia. Vista: Boosting 3d object detection via dual cross-view spatial attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8448-8457, 2022.

[12] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11963-11975, 2022.

[13] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Embracing single stride 3d object detector with sparse transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8458-8468, 2022.

[14] Lue Fan, Feng Wang, Naiyan Wang, and Zhao-Xiang Zhang. Fully sparse 3d object detection. Advances in Neural Information Processing Systems, 35:351-363, 2022.

[15] Lue Fan, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Fsd v2: Improving fully sparse 3d object detection with virtual voxels. arXiv preprint arXiv:2308.03755, 2023.

[16] Lue Fan, Yuxue Yang, Yiming Mao, Feng Wang, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. Once detected, never lost: Surpassing human performance in offline lidar based 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19820-19829, 2023.

[17] Lue Fan, Yuxue Yang, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Super sparse 3d object detection. IEEE transactions on pattern analysis and machine intelligence, 2023.

[18] Daniel Y Fu, Tri Dao, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint arXiv:2212.14052, 2022.

[19] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354-3361. IEEE, 2012.

<!-- Meanless: 11 -->


[20] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[21] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

[22] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems, 34:572-585, 2021.

[23] Tianrui Guan, Jun Wang, Shiyi Lan, Rohan Chandra, Zuxuan Wu, Larry Davis, and Dinesh Manocha. M3detr: Multi-representation, multi-scale, mutual-relation 3d object detection with transformers. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 772-782, 2022.

[24] Xu Han, Yuan Tang, Zhaoxuan Wang, and Xianzhi Li. Mamba3d: Enhancing local features for 3d point cloud analysis via state space model. arXiv preprint arXiv:2404.14966, 2024.

[25] Chenhang He, Ruihuang Li, Shuai Li, and Lei Zhang. Voxel set transformer: A set-to-set approach to 3d object detection from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8417-8427, 2022.

[26] Chenhang He, Hui Zeng, Jianqiang Huang, Xian-Sheng Hua, and Lei Zhang. Structure aware single-stage 3d object detection from point cloud. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11873-11882, 2020.

[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.

[28] David Hilbert and David Hilbert. Über die stetige abbildung einer linie auf ein flächenstück. Dritter Band: Analysis- Grundlagen der Mathematik- Physik Verschiedenes: Nebst Einer Lebens-geschichte, pages 1-2, 1935.

[29] Yihan Hu, Zhuangzhuang Ding, Runzhou Ge, Wenxin Shao, Li Huang, Kun Li, and Qiang Liu. Afdetv2: Rethinking the necessity of the second stage for object detection from point clouds. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 969-979, 2022.

[30] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697-12705, 2019.

[31] Jinyu Li, Chenxu Luo, and Xiaodong Yang. Pillarnext: Rethinking network designs for 3d object detection in lidar point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17567-17576, 2023.

[32] Yanwei Li, Yilun Chen, Xiaojuan Qi, Zeming Li, Jian Sun, and Jiaya Jia. Unifying voxel-based representation with transformer for 3d object detection. Advances in Neural Information Processing Systems, 35:18442-18455, 2022.

[33] Dingkang Liang, Xin Zhou, Xinyu Wang, Xingkui Zhu, Wei Xu, Zhikang Zou, Xiaoqing Ye, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. arXiv preprint arXiv:2402.10739, 2024.

[34] Jiuming Liu, Jinru Han, Lihao Liu, Angelica I Aviles-Rivero, Chaokang Jiang, Zhe Liu, and Hesheng Wang. Mamba4d: Efficient long-sequence point cloud video understanding with disentangled spatial-temporal state space models. arXiv preprint arXiv:2405.14338, 2024.

[35] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, and Yunfan Liu. Vmamba: Visual state space model. arXiv preprint arXiv:2401.10166, 2024.

[36] Ze Liu, Zheng Zhang, Yue Cao, Han Hu, and Xin Tong. Group-free 3d object detection via transformers. 2021 ieee. In CVF International Conference on Computer Vision (ICCV), pages 2929-2938, 2021.

<!-- Meanless: 12 -->


[37] Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, and Song Han. Flatformer: Flattened window attention for efficient point cloud transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1200-1211, 2023.

[38] Tao Lu, Xiang Ding, Haisong Liu, Gangshan Wu, and Limin Wang. Link: Linear kernel for lidar-based 3d perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1105-1115, 2023.

[39] Chuanyu Luo, Nuo Cheng, Sikun Ma, Han Li, Xiaohan Li, Shengguang Lei, and Pu Li. Lest: Large-scale lidar semantic segmentation with transformer. arXiv preprint arXiv:2307.09367, 2023.

[40] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. Advances in neural information processing systems, 29, 2016.

[41] Jiageng Mao, Yujing Xue, Minzhe Niu, Haoyue Bai, Jiashi Feng, Xiaodan Liang, Hang Xu, and Chunjing Xu. Voxel transformer for 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3164-3173, 2021.

[42] Mohamed F Mokbel, Walid G Aref, and Ibrahim Kamel. Analysis of multi-dimensional space-filling curves. GeoInformatica, 7:179-209, 2003.

[43] Guy M Morton. A computer oriented geodetic data base and a new technique in file sequencing. 1966.

[44] Yong-Joo Oh and Yoshio Watanabe. Development of small robot for home floor cleaning. In Proceedings of the 41st SICE Annual Conference. SICE 2002., volume 5, pages 3222-3223. IEEE, 2002.

[45] Jack A Orenstein. Spatial query processing in an object-oriented database system. In Proceedings of the 1986 ACM SIGMOD international conference on Management of data, pages 326-336, 1986.

[46] Youngmin Park, Vincent Lepetit, and Woontack Woo. Multiple 3d object tracking for augmented reality. In 2008 7th IEEE/ACM International Symposium on Mixed and Augmented Reality, pages 117-120. IEEE, 2008.

[47] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9277-9286, 2019.

[48] Charles R Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J Guibas. Frustum pointnets for 3d object detection from rgb-d data. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 918-927, 2018.

[49] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652-660, 2017.

[50] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017.

[51] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Fcaf3d: Fully convolutional anchor-free 3d object detection. In European Conference on Computer Vision, pages 477-493. Springer, 2022.

[52] Guangsheng Shi, Ruifeng Li, and Chao Ma. Pillarnet: Real-time and high-performance pillar-based 3d object detection. In European Conference on Computer Vision, pages 35-52. Springer, 2022.

[53] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10529-10538, 2020.

<!-- Meanless: 13 -->


[54] Shaoshuai Shi, Li Jiang, Jiajun Deng, Zhe Wang, Chaoxu Guo, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn++: Point-voxel feature set abstraction with local vector representation for 3d object detection. International Journal of Computer Vision, 131(2):531-551, 2023.

[55] Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. Pointrcnn: 3d object proposal generation and detection from point cloud. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 770-779, 2019.

[56] Shaoshuai Shi, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. From points to parts: 3d object detection from point cloud with part-aware and part-aggregation network. IEEE transactions on pattern analysis and machine intelligence, 43(8):2647-2664, 2020.

[57] John Skilling. Programming the hilbert curve. In AIP Conference Proceedings, volume 707, pages 381-387. American Institute of Physics, 2004.

[58] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022.

[59] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446-2454, 2020.

[60] Pei Sun, Mingxing Tan, Weiyue Wang, Chenxi Liu, Fei Xia, Zhaoqi Leng, and Dragomir Anguelov. Swformer: Sparse window transformer for 3d object detection in point clouds. In European Conference on Computer Vision, pages 426-442. Springer, 2022.

[61] Pei Sun, Weiyue Wang, Yuning Chai, Gamaleldin Elsayed, Alex Bewley, Xiao Zhang, Cristian Sminchiesscu, and Dragomir Anguelov. Rsn: Range sparse net for efficient, accurate lidar 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5725-5734, 2021.

[62] OpenPCDet Development Team. Openpcdet: An open-source toolbox for 3d object detection from point clouds. https://github.com/open-mmlab/OpenPCDet.2020.

[63] Zhi Tian, Xiangxiang Chu, Xiaoming Wang, Xiaolin Wei, and Chunhua Shen. Fully convolutional one-stage 3d object detection on lidar range images. Advances in Neural Information Processing Systems, 35:34899-34911, 2022.

[64] Haiyang Wang, Chen Shi, Shaoshuai Shi, Meng Lei, Sen Wang, Di He, Bernt Schiele, and Liwei Wang. Dsvt: Dynamic sparse voxel transformer with rotated sets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13520-13529, 2023.

[65] Peng-Shuai Wang. Octformer: Octree-based transformers for 3d point clouds. arXiv preprint arXiv:2305.03045, 2023.

[66] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1-12, 2019.

[67] Zicheng Wang, Zhenghao Chen, Yiming Wu, Zhen Zhao, Luping Zhou, and Dong Xu. Pointramba: A hybrid transformer-mamba framework for point cloud analysis. arXiv preprint arXiv:2405.15463, 2024.

[68] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler, faster, stronger. arXiv preprint arXiv:2312.10035, 2023.

[69] Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.

[70] Honghui Yang, Zili Liu, Xiaopei Wu, Wenxiao Wang, Wei Qian, Xiaofei He, and Deng Cai. Graph r-cnn: Towards accurate 3d object detection with semantic-decorated local graph. In European Conference on Computer Vision, pages 662-679. Springer, 2022.

<!-- Meanless: 14 -->


[71] Zetong Yang, Yanan Sun, Shu Liu, and Jiaya Jia. 3dssd: Point-based 3d single stage object detector. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11040-11048, 2020.

[72] Zetong Yang, Yin Zhou, Zhifeng Chen, and Jiquan Ngiam. 3d-man: 3d multi-frame attention network for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1863-1872, 2021.

[73] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Center-based 3d object detection and tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11784-11793, 2021.

[74] Gang Zhang, Chen Junnan, Guohuan Gao, Jianmin Li, and Xiaolin Hu. Hednet: A hierarchical encoder-decoder network for 3d object detection in point clouds. Advances in Neural Information Processing Systems, 36, 2024.

[75] Tao Zhang, Xiangtai Li, Haobo Yuan, Shunping Ji, and Shuicheng Yan. Point could mamba: Point cloud learning via state space model. arXiv preprint arXiv:2403.00762, 2024.

[76] Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Rui Zhang, Keyi Liu, Yeqi Luo, and Ying He. 3dmambaipf: A state space model for iterative point cloud filtering via differentiable rendering. arXiv preprint arXiv:2404.05522, 2024.

[77] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4490-4499, 2018.

[78] Zixiang Zhou, Xiangchen Zhao, Yu Wang, Panqu Wang, and Hassan Foroosh. Centerformer: Center-based transformer for 3d object detection. In European Conference on Computer Vision, pages 496-513. Springer, 2022.

[79] Benjin Zhu, Zhe Wang, Shaoshuai Shi, Hang Xu, Lanqing Hong, and Hongsheng Li. Conquer: Query contrast voxel-detr for 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9296-9305, 2023.

[80] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024.

<!-- Meanless: 15 -->