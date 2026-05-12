<!-- Meanless: arXiv:2407.18232v1 [cs.CV] 25 Jul 2024 -->

# LION: Linear Group RNN for 3D Object Detection in Point Clouds

Zhe Liu ${}^{1 * }$ , Jinghua Hou ${}^{1 * }$ , Xinyu Wang ${}^{1 * }$ , Xiaoqing Ye ${}^{3}$ , Jingdong Wang ${}^{3}$ , Hengshuang Zhao ${}^{2}$ , Xiang Bai ${}^{1 \dagger  }$

${}^{1}$ Huazhong University of Science and Technology

${}^{2}$ The University of Hong Kong ${}^{3}$ Baidu Inc.,China

https://happinesslz.github.io/projects/LION

## Abstract

The benefit of transformers in large-scale 3D point cloud perception tasks, such as 3D object detection, is limited by their quadratic computation cost when modeling long-range relationships. In contrast, linear RNNs have low computational complexity and are suitable for long-range modeling. Toward this goal, we propose a simple and effective window-based framework built on LInear grOup RNN (i.e., perform linear RNN for grouped features) for accurate 3D object detection, called LION. The key property is to allow sufficient feature interaction in a much larger group than transformer-based methods. However, effectively applying linear group RNN to 3D object detection in highly sparse point clouds is not trivial due to its limitation in handling spatial modeling. To tackle this problem, we simply introduce a 3D spatial feature descriptor and integrate it into the linear group RNN operators to enhance their spatial features rather than blindly increasing the number of scanning orders for voxel features. To further address the challenge in highly sparse point clouds, we propose a 3D voxel generation strategy to densify foreground features thanks to linear group RNN as a natural property of auto-regressive models. Extensive experiments verify the effectiveness of the proposed components and the generalization of our LION on different linear group RNN operators including Mamba, RWKV, and RetNet. Furthermore, it is worth mentioning that our LION-Mamba achieves state-of-the-art on Waymo, nuScenes, Argoverse V2, and ONCE datasets. Last but not least, our method supports kinds of advanced linear RNN operators (e.g., RetNet, RWKV, Mamba, xLSTM and TTT) on small but popular KITTI dataset for a quick experience with our linear RNN-based framework.

## 1 Introduction

3D object detection serves as a fundamental technique in 3D perception and is widely used in navigation robots and self-driving cars. Recently, transformer-based [53] feature extractors have made significant progress in general tasks of Natural Language Processing (NLP) and 2D vision by flexibly modeling long-range relationships. To this end, some researchers have made great efforts to transfer the success of transformers to 3D object detection. Specifically, to reduce the computation costs, SST [15] and SWFormer [50] divide point clouds into pillars and implement window attention for pillar feature interaction in a local 2D window. Considering some potential information loss of the pillar-based manners along the height dimension, DSVT-Voxel [57] further adopts voxel-based formats and implements set attention for voxel feature interaction in a limited group size.

---

<!-- Footnote -->

*Equal contribution.

${}^{ \dagger  }$ Corresponding author.

<!-- Footnote -->

---

<!-- Meanless: Preprint. Under review. -->


<!-- Media -->

<!-- figureText: Small Groups<br>Waymo mAP/L1<br>82.1<br>Argoverse V2 mAP<br>Waymo mAP/L2<br>Transformer<br>80.3<br>80.1<br>74.4<br>71.7<br>nuScenes mAP (val)<br>Waymo mAPH/L1<br>75.9<br>Window<br>66.4<br>(b) DSVT<br>59.2<br>68.2<br>N/A<br>65.8<br>nuScenes NDS (val)<br>58.0<br>Waymo mAPH/L2<br>Large Groups<br>Linear Group RNN<br>67.7<br>68.4<br>72.0<br>66.6<br>nuScenes mAP (test)<br>ONCE mAP<br>73.9<br>nuScenes NDS (test)<br>Window<br>☐LION<br>☐CenterPoint<br>☐DSVT<br>☐HEDNet<br>(a) Performance comparison<br>(c) LION (Ours) -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_1.jpg?x=314&y=202&w=1170&h=482&r=0"/>

Figure 1: (a) Comparison of different 3D backbones in terms of detection performance on Waymo [49], nuScenes [4], Argoverse V2 [59] and ONCE [34] datasets. Here, we adopt Mamba [22] as the default operator of our LION. Besides, we present the simplified schematic of DSVT (b) [57] and our LION (c) for implementing feature interaction in 3D backbones.

<!-- Media -->

Although the above methods have achieved some success in 3D detection, they perform self-attention for pillar or voxel feature interaction with only a small group size due to computational limitations, locking the potential of transformers for modeling long-range relationships. Moreover, it is worth noting that modeling long-range relationships can benefit from large datasets, which will be important for achieving foundational models in 3D perception tasks in the future. Fortunately, in the field of large language models (LLM) and 2D perception tasks, some representative linear RNN operators such as Mamba [22] and RWKV [37] with linear computational complexity have achieved competitive performance with transformers, especially for long sequences. Therefore, a question naturally arises: can we perform long-range feature interaction in larger groups at a lower computation cost based on linear RNNs in 3D object detection?

To this end, we propose a window-based framework based on LInear grOup RNN (i.e., perform linear RNN for grouped features in a window-based framework) termed LION for accurate 3D object detection in point clouds. Different from the existing method DSVT (b) in Figure 1, our LION (c) could support thousands of voxel features to interact with each other in a large group for establishing the long-range relationship. Nevertheless, effectively adopting linear group RNN to construct a proper 3D detector in highly sparse point cloud scenes remains challenging for capturing the spatial information of objects. Concretely, linear group RNN requires sequential features as inputs. However, converting voxel features into sequential features may result in the loss of spatial information (e.g., two features that are close in 3D spatial position might be very far in this 1D sequence). Therefore, we propose a simple 3D spatial feature descriptor and decorate the linear group RNN operators with it, thus compensating for the limitations of linear group RNN in 3D local spatial modeling.

Furthermore, to enhance feature representation in highly sparse point clouds, we present a new 3D voxel generation strategy based on linear group RNN to densify foreground features. A common manner of addressing this is to add an extra branch to distinguish the foregrounds, as seen in previous methods [50, 17, 67]. However, this solution is relatively complex and rarely used in 3D backbone due to its lack of structural elegance. Instead, we simply choose the high response of the feature map in the 3D backbone as the areas for voxel generation. Subsequently, the auto-regressive property of linear group RNN can be effectively employed to generate voxel features.

Finally, as shown in Figure 1 (a), we compare LION with the existing representative methods. We can clearly observe that our LION achieves state-of-the-art on a board autonomous datasets in terms of detection performance. To summarize, our contributions are as follows: 1) We propose a simple and effective window-based 3D backbone based on the linear group RNN named LION to allow long-range feature interaction. 2) We introduce a simple 3D spatial feature descriptor and integrate it with the linear group RNN, compensating for the lack of capturing 3D local spatial information. 3) We provide a new 3D voxel generation strategy to densify foreground features, producing a more discriminative feature representation in highly sparse point clouds. 4) We verify the generalization of our LION with different linear group RNN mechanisms (e.g., Mamba, RWKV, RetNet). In particular,

<!-- Meanless: 2 -->


<!-- Media -->

<!-- figureText: $\left( {H,W,\frac{D}{2}}\right)$<br>$\left( {H,W,\frac{D}{{2}^{N}}}\right)$<br>Voxelization<br>LION Block #1<br>Generation Voxel<br>Voxel<br>LION Block #N<br>Generation Voxel<br>Merging Voxel<br>Backbone BEV<br>Head<br>Point Clouds<br>LION 3D Backbone -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_2.jpg?x=312&y=204&w=1172&h=204&r=0"/>

Figure 2: The illustration of LION, which mainly consists of several LION blocks, each paired with a voxel generation for feature enhancement and a voxel merging for down-sampling features along the height dimension. $\left( {H,W,D}\right)$ indicates the shape of the 3D feature map,where $H,W$ ,and $D$ are the length,width,and height of the 3D feature map along the X-axis,Y-axis,and Z-axis. $N$ is the number of LION blocks. In LION, we first convert point clouds to voxels and partition these voxels into a series of equal-size groups. Then, we feed these grouped features into LION 3D backbone to enhance their feature representation. Finally, these enhanced features are fed into a BEV backbone and a detection head for final 3D detection.

<!-- Media -->

our LION-Mamba achieves state-of-the-art on challenging Waymo [49], nuScenes [4], Argoverse V2 [59], and ONCE [34] dataset, which further illustrates the superiority of LION.

## 2 Related Work

3D Object Detection in Point Clouds. 3D object detectors in point clouds can be roughly divided into point-based and voxel-based. For point-based methods [6, 65, 39, 10, 30, 36, 24, 47, 69, 64, 40, 62, 5], they usually sample point clouds and adopt point encoder [41, 42] to directly extract point features. However, the point sampling and grouping utilized by point-based methods is time-consuming. To avoid these problems, voxel-based methods [13, 12, 31, 45, 46, 48, 23, 56, 60, 66, 61, 67] convert the input irregular point clouds into regular 3D voxels and then extract 3D features by 3D sparse convolution. Although these methods achieve promising performance, they are still limited by the local receptive field of 3D convolution. Therefore, some methods [8, 33] adopt the large kernel to enlarge the receptive field and achieve better performance.

Linear RNN. Recurrent Neural Networks (RNNs) are initially developed to address problems in Natural Language Processing (NLP), such as time series prediction and speech recognition, by effectively capturing temporal dependencies in sequential data. Recently, to overcome the quadratic computational complexity of transformers, significant advancements have been made in time-parallelizable data-dependent RNNs (called linear RNNs in this paper) [43, 35, 37, 38, 52, 11, 63, 22, 51, 3]. These models retain linear complexity while offering efficient parallel training capabilities, allowing their performance to match or even surpass that of transformers. Due to their scalability and efficiency, linear RNNs are poised to play an increasingly important role in various fields and some works [14, 1, 28] have applied linear RNNs to 2D/3D vision filed. In this paper, we aim to further extend linear RNNs to 3D object detection tasks thanks to their long-range relationship modeling capabilities.

Transformers in 3D Object Detection. Transformer [53] has achieved great success in many tasks, motivating numerous works to adopt attention mechanisms in 3D object detection to achieve better performance. However, the application of transformers is non-trivial in large-scale point clouds. Many works $\left\lbrack  {{16},{50},{32},{57}}\right\rbrack$ apply transformers to extract features by partitioning pillars or voxels into several groups based on local windows. Although these approaches achieve promising performance, they usually adopt small groups for feature interaction due to the quadratic computational complexity of transformers, hindering them from capturing long-range dependencies in 3D space. In contrast, we propose a simple and effective framework based on linear RNNs named LION to achieve long-range feature interaction for accurate 3D object detection thanks to their linear computational complexity.

## 3 Method

Due to computational limitations, some transformer-based methods [15, 57, 32] usually convert features into pillars or group small size of voxel features to interact with each other within small groups, limiting the advantages of transformers in long-range modeling. More recently, some linear RNN operators [22, 37, 52] that maintain linear complexity with the length of the input sequence are proposed to model long-range feature interaction. More importantly, the linear RNN operators such as Mamba [22] and RWKV [37] have even shown comparable performance with transformers in LLM thanks to their low computation cost in long-range feature interaction. This further motivates us to adopt linear RNNs to construct a 3D detector for long-range modeling, which might be meaningful for the unified multi-modal large model in 3D perception in the near future. However, effectively applying linear group RNN to 3D object detection is challenging and rarely explored due to the lack of strong spatial modeling in highly sparse point cloud scenes. Next, we will introduce our solution.

<!-- Meanless: 3 -->


<!-- Media -->

<!-- figureText: ${\overline{2}}^{ \times }$<br>1×<br>LION<br>⑤<br>Merging Voxel<br>Layer LION<br>Merging Voxel<br>LION<br>Expanding Voxel<br>Layer LION<br>Expanding Voxel<br>(a) LION Block<br>3D Sub-Manifold Convolution<br>X-axis Partition<br>Voxel<br>Voxel<br>Merging<br>Expanding<br>Linear Group RNN<br>LayerNorm<br>(b) Voxel Merging and Expanding<br>Y-axis Partition<br>GELU<br>$\oplus$<br>Add<br>⑤<br>3D Spatial Feature Descriptor<br>Linear Group RNN<br>Index Mappings<br>Inversed Index Mappings<br>(c) LION Layer<br>(d)3D Spatial Feature Descriptor -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_3.jpg?x=334&y=203&w=1113&h=519&r=0"/>

Figure 3: (a) shows the structure of LION block, which involves four LION layers, two voxel merging operations, two voxel expanding operations, and two 3D spatial feature descriptors. Here, $1 \times  ,\frac{1}{2} \times$ ,and $\frac{1}{4} \times$ indicate the resolution of 3D feature map as $\left( {H,W,D}\right) ,\left( {H/2,W/2,D/2}\right)$ and $\left( {H/4,W/4,D/4}\right)$ ,respectively. (b) is the process of voxel merging for voxel down-sampling and voxel expanding for voxel up-sampling. (c) presents the structure of LION layer. (d) shows the details of the 3D spatial feature descriptor.

<!-- Media -->

### 3.1 Overview

In this paper, we propose a simple and effective window-based framework based on LInear grOup RNN (i.e., perform linear RNN for grouped features in a window-based framework) named LION, which can group thousands of voxels (dozens of times more than the number of previous methods [15, 57, 32]) for feature interaction. The pipeline of our LION is presented in Figure 2. LION consists of a 3D backbone, a BEV backbone, and a detection head, maintaining a consistent pipeline with most voxel-based 3D detectors [60, 57, 66]. In this paper, our contribution lies in the design of the 3D backbone based on linear group RNN. In the following, we will present the details of our proposed 3D backbone,which includes $N$ LION blocks for long-range feature interaction, $N$ voxel generation operations for enhancing feature representation in sparse point clouds,and $N$ voxel merging operations for gradually down-sampling features in height.

3D Sparse Window Partition. Our LION is a window-based 3D detector. Thus, before feeding voxel features into our LION block, we need to implement a 3D sparse window partition to group them for feature interaction. Specifically, we first convert point clouds into voxels with the total number of $L$ . Then,we divide these voxels into non-overlapping 3D windows with the shape of $\left( {{T}_{x},{T}_{y},{T}_{z}}\right)$ , where ${T}_{x},{T}_{y}$ and ${T}_{z}$ denote the length,width,and height of the window along the X-axis,Y-axis, and Z-axis. Next, we sort voxels along the X-axis for the X-axis window partition and along the Y-axis for the Y-axis window partition, respectively. Finally, to save computation cost, we adopt the equal-size grouping manner in FlatFormer [32] instead of the classic equal-window grouping manner in SST [15]. That is,we partition sorted voxels into groups with equal size $K$ rather than windows of equal shapes for feature interaction. Due to the quadratic computational complexity of transformers, previous transformer-based methods $\left\lbrack  {{15},{57},{32}}\right\rbrack$ only achieve feature interaction using a small group size. In contrast,we adopt a much larger group size $K$ to obtain long-range feature interaction thanks to the linear computational complexity of the linear group RNN operators.

<!-- Meanless: 4 -->


### 3.2 LION Block

The LION block is the core component of our approach, which involves LION layer for long-range feature interaction, 3D spatial feature descriptor for capturing local 3D spatial information, voxel merging for feature down-sampling and voxel expanding for feature up-sampling, as shown in Figure 3 (a). Besides, LION block is a hierarchical structure to better extract multi-scale features due to the gap of different 3D objects in size. Next, we introduce each part of LION block.

LION Layer. In LION block, we apply LION layer to model a long-range relationship among grouped features with the help of the linear group RNN operator. Specifically, as shown in Figure 3 (c), we provide the structure of LION layer, which is composed of two linear group RNN operators. The first one is used to perform long-range feature interaction based on the X-axis window partition and the second one can extract long-range feature information based on the Y-axis window partition. Taking advantage of two different window partitions, LION layer can obtain more sufficient feature interaction, producing more discriminative feature representation.

3D Spatial Feature Descriptor. Although linear RNNs have the advantages of long-range modeling with low computation cost, it is not ignorable that the spatial information might be lost when input voxel features are flattened into 1D sequential features. For example, as shown in Figure 4, there are two adjacent features (i.e., indexed as 01 and 34) in 3D space. However, after they are flattened into 1D sequential features, the distance between them in 1D space is very far. We regard this phenomenon as a loss of 3D spatial information. To tackle this problem, an available manner is to increase the number of scan orders for voxel features such as VMamba [29]. However, the order of scanning is too hand-designed. Besides, as the scanning orders increase, the corresponding computation cost also increases significantly. Therefore, it is not appropriate in large-scale sparse 3D point clouds to adopt this manner. As shown in Figure 3 (d), we introduce a 3D spatial feature descriptor, which consists of a 3D sub-manifold convolution, a LayerNorm layer, and a GELU activation function. Naturally, we can leverage the 3D spatial feature descriptor to provide rich 3D local position-aware information for the LION layer. Besides, we place the 3D spatial feature descriptor before the voxel merging to reduce spatial information loss in the process of voxel merging. We provide the corresponding experiment in our appendix.

<!-- Media -->

<!-- figureText: X<br>01<br>02<br>32)<br>33<br>34)<br>64)<br>65<br>66<br>96<br>у -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_4.jpg?x=1128&y=700&w=357&h=289&r=0"/>

Figure 4: The illustration of spatial information loss when flattening into 1D sequences. For example, there are two adjacent voxels in spatial position (indexed as 01 and 34) but are far in the 1D sequences along the X order.

<!-- Media -->

Voxel Merging and Voxel Expanding. To enable the network to obtain multi-scale features, our LION adopts a hierarchical feature extraction structure. To achieve this, we need to perform feature down-sampling and up-sampling operations in highly sparse point clouds. However, it is worth mentioning that we cannot simply apply max or average pooling or up-sampling operations as in 2D images since 3D point clouds possess irregular data formats. Therefore, as shown in Figure 3 (b), we adopt voxel merging for feature down-sampling and voxel expanding for feature up-sampling in highly sparse point clouds. Specifically, for voxel merging, we calculate the down-sampled index mappings to merge voxels. In voxel expanding, we up-sample the down-sampled voxels by the corresponding inversed index mappings.

### 3.3 Voxel Generation

Considering the challenge of feature representation in highly sparse point clouds and the potential information loss of implementing voxel merging in Figure 2, we propose a voxel generation strategy to address these issues with the help of the auto-regressive capacity of the linear group RNN.

Distinguishing Foreground Voxels without Supervision. In voxel generation, the first challenge is identifying which regions of voxel features need to be generated. Different from previous methods [50, 17, 67] that employ some supervised information based on well-learned BEV features to obtain the foreground region for feature diffusion. However, these approaches may be unsuitable for a 3D backbone and may even compromise its elegance. Interestingly, we notice that the corresponding high values of feature responses along the channel dimension in the 3D backbone (Refer to the visualization in Figure 6 in our appendix) are usually the foregrounds. Therefore, we compute the feature response ${F}_{i}^{ * }$ for the output feature ${F}_{i}$ of the ${i}^{th}$ LION block,where $i = 1,2,\ldots ,N$ indicates the index of LION block in the 3D backbone. Thus, this above process can be formulated as:

<!-- Meanless: 5 -->


$$
{F}_{i}^{ * } = \frac{1}{C}\mathop{\sum }\limits_{{j = 0}}^{C}{F}_{i}^{j} \tag{1}
$$

<!-- Media -->

<!-- figureText: 十<br>$\left\lbrack {-1,1,0}\right\rbrack$<br>$\left\lbrack {1,1,0}\right\rbrack$<br>LION Block<br>$\left\lbrack {-1, - 1,0}\right\rbrack$<br>Input Voxels<br>Voxel Diffusion<br>Voxel Generation based on Auto-regressive Model<br>Output Voxels<br>Input Voxel<br>Diffused Voxel<br>Foreground Voxel<br>Generated Voxel -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_5.jpg?x=312&y=207&w=1169&h=303&r=0"/>

Figure 5: The details of voxel generation. For input voxels, we first select the foreground voxels and diffuse them along different directions. Then, we initialize the corresponding features of the diffused voxels as zeros and utilize the auto-regressive ability of the following LION block to generate diffused features. Note that we do not present the voxel merging here for simplicity.

<!-- Media -->

where $C$ is the channel dimension of ${F}_{i}$ . Next,we sort the feature responses ${F}_{i}^{ * }$ in descending order and select the corresponding Top- $m$ voxels as the foregrounds from the total number $L$ of non-empty voxels,where $m = r * L$ and $r$ is the ratio of foregrounds. This process can be computed as:

$$
{F}_{m} = {\operatorname{Top}}_{m}\left( {F}_{i}^{ * }\right) , \tag{2}
$$

where ${\operatorname{Top}}_{m}\left( {F}_{i}^{ * }\right)$ means selecting Top- $m$ voxel features from ${F}_{i}^{ * }.{F}_{m}$ are the selected foreground features, which will serve for the subsequent voxel generation.

Voxel Generation with Auto-regressive Property. The previous method [50] adopts a K-NN manner to obtain generated voxel features based on their K-NN features, which might be sub-optimal to enhance feature representation due to the redundant features and the limited receptive field. Fortunately, the linear RNN is well-suited for auto-regressive tasks in addition to its advantage of handling long sequences. Therefore, we leverage the auto-regressive property of linear RNN to effectively generate the new voxel features by performing sufficient feature interaction with other voxel features in a large group. Specifically, for convenience, we define the corresponding coordinates of selected foreground voxel features ${F}_{m}$ as ${P}_{m}$ . As shown in Figure 5,we first obtain diffused voxels by diffusing ${P}_{m}$ with four different offsets (i.e., $\left\lbrack  {-1, - 1,0}\right\rbrack  ,\left\lbrack  {1,1,0}\right\rbrack  ,\left\lbrack  {1, - 1,0}\right\rbrack$ ,and $\left\lbrack  {-1,1,0}\right\rbrack$ ) along the X-axis, Y-axis, and Z-axis, respectively. Then, we initialize the corresponding features of diffused voxels by all zeros. Next,we concatenate the output feature ${F}_{i}$ of the ${i}^{th}$ LION block with the initialized voxel features,and feed them into the subsequent ${\left( i + 1\right) }^{th}$ LION block. Finally, thanks to the auto-regressive ability of the LION block, the diffused voxel features can be effectively generated based on other voxel features in large groups. This process can be formulated as:

$$
{F}_{p} = {F}_{i} \oplus  {F}_{\left\lbrack  -1, - 1,0\right\rbrack  } \oplus  {F}_{\left\lbrack  1,1,0\right\rbrack  } \oplus  {F}_{\left\lbrack  1, - 1,0\right\rbrack  } \oplus  {F}_{\left\lbrack  -1,1,0\right\rbrack  }, \tag{3}
$$

$$
{F}_{p}^{\prime } = \operatorname{Block}\left( {F}_{p}\right) , \tag{4}
$$

where ${F}_{\left\lbrack  x,y,z\right\rbrack  }$ denotes the initialized voxel features with diffused offsets of x,y,and z along the X-axis, Y-axis, and Z-axis. The $\oplus$ and Block denote the concatenation and LION block respectively.

## 4 Experiments

### 4.1 Datasets and Evaluation Metrics

Waymo Open Dataset. Waymo Open dataset (WOD) [49] is a well-known benchmark for large-scale outdoor 3D perception, comprising 1150 scenes which are divided into 798 scenes for training, 202 scenes for validation, and 150 scenes for testing. Each scene includes about 200 frames, covering a perception range of ${150m} \times  {150m}$ . For evaluation metrics,WOD employs 3D mean Average Precision (mAP) and mAP weighted by heading accuracy (mAPH), each divided into two difficulty levels: L1 is for objects detected with more than five points and L2 is for those at least one point. nuScenes Dataset. nuScenes [4] is a popular outdoor 3D perception benchmark with a perception range of up to 50 meters. Each frame in the scene is annotated with $2\mathrm{\;{Hz}}$ . The dataset includes 1000 scenes, which is divided into 750 scenes for training, 150 scenes for validation, and 150 scenes for testing. nuScenes adopts mean Average Precision (mAP) and the NuScenes Detection Score (NDS) as evaluation metrics.

<!-- Meanless: 6 -->


Argoverse V2 Dataset. Argoverse V2 [59] is a popular outdoor 3D perception benchmark with a long-range perception of up to 200 meters. It contains 1000 sequences in total, 700 for training, 150 for validation, and 150 for testing. Each frame in the scene is annotated with 10Hz. For the evaluation metric, Argoverse v2 adopts a similar mean Average Precision (mAP) metric with nuScenes [4].

ONCE Dataset. ONCE [34] is another representative autonomous driving dataset, which consists of 5000, 3000, and 8000 frames for training, validation, and testing set, respectively. Each frame is annotated with 5 classes (Car, Bus, Truck, Pedestrian, and Cyclist). Besides, ONCE merges the car, bus, and truck class into a super-class called vehicle following WOD [49]. For the detection metric, ONCE extends [19] by taking the object orientations into special consideration and evaluating the final performance by mAP for three classes.

### 4.2 Implementation Details

Network Architecture. In our LION, we provide three representative linear RNN operators (i.e., Mamba [22], RWKV [37], and RetNet [52]). Each of operator adopts a bi-directional structure to better capture 3D geometric information inspired by [21]. On WOD, we keep the same channel dimension $C = {64}$ for all LION blocks in LION-Mamba,LION-RWKV,and LION-RetNet. For the large version of LION-Mamba-L, we set $C = {128}$ . We follow DSVT-Voxel [57] to set the grid size as $\left( {{0.32}\mathrm{\;m},{0.32}\mathrm{\;m},{0.1875}\mathrm{\;m}}\right)$ . The number of LION blocks $N$ is set to 4 . For these four LION blocks,the window sizes $\left( {{T}_{x},{T}_{y},{T}_{z}}\right)$ are set to $\left( {{13},{13},{32}}\right) ,\left( {{13},{13},{16}}\right) ,\left( {{13},{13},8}\right)$ ,and $\left( {{13},{13},4}\right)$ , and the corresponding group sizes $K$ are 4096,2048,1024,512,respectively. Besides,we adopt the same center-based detection head and loss function as DSVT [57] for fair comparison. In the voxel generation,we set the ratio $r = {0.2}$ to balance the performance and computation cost. For the nuScenes dataset, we replace DSVT [57] 3D backbone with our LION 3D backbone except for changing the grid size to $\left( {{0.32m},{0.32m},{0.15m}}\right)$ . For the Argoverse V2 dataset,we replace the 3D backbones of VoxelNext [9] or SAFDNet [67] with our LION 3D backbone except for setting the grid size to $\left( {{0.4m},{0.4m},{0.25m}}\right)$ . Moreover,it is noted that we only add three extra LION layers to further enhance the 3D backbone features, rather than applying the BEV backbone to obtain the dense BEV features. For the ONCE dataset, we replace SECOND [60] 3D backbone with our LION 3D backbone except for adopting the grid size as $\left( {{0.4m},{0.4m},{0.25m}}\right)$ .

Training Process. On the WOD, we adopt the same point cloud range, data augmentations, learning rate, and optimizer as the previous method [57]. We train our model 24 epochs with a batch size of 16 on 8 NVIDIA Tesla V100 GPUs. Besides, we utilize the fade strategy [55] to achieve better performance in the last epoch. For the nuScenes dataset, we adopt the same point cloud range, data augmentations, and optimizer as previous method [2]. Moreover, we find that LION converges faster than previous methods on nuScenes dataset. Therefore, we only train our model for 36 epochs without CBGS [73]. The learning rate and batch size are set to 0.003 and 16, respectively. It is worth noting that the CBGS strategy extends training iterations about 4.5 times, which means that our training iterations are much fewer than previous methods [2] (i.e., 20 epochs with CBGS). For the Argoverse V2 dataset and the ONCE dataset, we adopt the same training process with SAFDNet [67] and SECOND [60], respectively.

### 4.3 Main Results

In this section, we provide a board comparison of our LION with existing methods on WOD, nuScenes, Argoverse V2 and ONCE datasets for 3D object detection. Furthermore, in the section A. 1 of our appendix, we present more types of linear RNN operators (e.g., RetNet, RWKV, Mamba, xLSTM, and TTT) based on our LION framework for 3D detection on a small but popular dataset KITTI [20] for a quick experience.

Results on WOD. To illustrate the superiority of our LION, we provide the comparison with existing representative methods on the WOD in Table 1. Here, we also conduct the experiments on our LION with different linear group RNN operators, including LION-Mamba, LION-RWKV and LION-RetNet.

<!-- Meanless: 7 -->


<!-- Media -->

Table 1: Performances on the Waymo Open Dataset validation set (train with ${100}\%$ training data). $\ddagger$ denotes the two-stage method. Bold denotes the best performance of all methods. "-L" means we double the dimension of channels in LION 3D backbone. RNN denotes the linear RNN operator. All results are presented with single-frame input, no test-time augmentation, and no model ensembling.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Present at</td><td rowspan="2">Operator</td><td colspan="2">Vehicle 3D AP/APH</td><td colspan="2">Pedestrian 3D AP/APH</td><td colspan="2">Cyclist 3D AP/APH</td><td rowspan="2">mAP/mAPH <br> L2</td></tr><tr><td>L1</td><td>L2</td><td>L1</td><td>L2</td><td>L1</td><td>L2</td></tr><tr><td>SECOND [60]</td><td>Sensors 18</td><td></td><td>72.3/71.7</td><td>63.9/63.3</td><td>68.7/58.2</td><td>60.7/51.3</td><td>60.6/59.3</td><td>58.3/57.0</td><td>61.0/57.2</td></tr><tr><td>PointPillars [26]</td><td>CVPR 19</td><td></td><td>72.1/71.5</td><td>63.6/63.1</td><td>70.6/56.7</td><td>62.8/50.3</td><td>64.4/62.3</td><td>61.9/59.9</td><td>62.8/57.8</td></tr><tr><td>CenterPoint [66]</td><td>CVPR 21</td><td></td><td>74.2/73.6</td><td>66.2/65.7</td><td>76.6/70.5</td><td>68.8/63.2</td><td>72.3/71.1</td><td>69.7/68.5</td><td>68.2/65.8</td></tr><tr><td>PV-RCNN‡ [45]</td><td>CVPR 20</td><td></td><td>78.0/77.5</td><td>69.4/69.0</td><td>79.2/73.0</td><td>70.4/64.7</td><td>71.5/70.3</td><td>69.0/67.8</td><td>69.6/67.2</td></tr><tr><td>PillarNet-34 [44]</td><td>ECCV 22</td><td></td><td>79.1/78.6</td><td>70.9/70.5</td><td>80.6/74.0</td><td>72.3/66.2</td><td>72.3/71.2</td><td>69.7/68.7</td><td>71.0/68.5</td></tr><tr><td>FSD: [17]</td><td>NeurIPS 22</td><td></td><td>79.2/78.8</td><td>70.5/70.1</td><td>82.6/77.3</td><td>73.9/69.1</td><td>77.1/76.0</td><td>74.4/73.3</td><td>72.9/70.8</td></tr><tr><td>AFDetV2 [25]</td><td>AAAI 22</td><td></td><td>77.6/77.1</td><td>69.7/69.2</td><td>80.2/74.6</td><td>72.2/67.0</td><td>73.7/72.7</td><td>71.0/70.1</td><td>71.0/68.8</td></tr><tr><td>PillarNeXt [27]</td><td>CVPR 23</td><td>Sparse Conv</td><td>78.4/77.9</td><td>70.3/69.8</td><td>82.5/77.1</td><td>74.9/69.8</td><td>73.2/72.2</td><td>70.6/69.6</td><td>71.9/69.7</td></tr><tr><td>VoxelNext [9]</td><td>CVPR 23</td><td></td><td>78.2/77.7</td><td>69.9/69.4</td><td>81.5/76.3</td><td>73.5/68.6</td><td>76.1/74.9</td><td>73.3/72.2</td><td>72.2/70.1</td></tr><tr><td>CenterFormer[72]</td><td>ECCV 22</td><td></td><td>75.0/74.4</td><td>69.9/69.4</td><td>78.6/73.0</td><td>73.6/68.3</td><td>72.3/71.3</td><td>69.8/68.8</td><td>71.1/68.9</td></tr><tr><td>PV-RCNN++1 [46]</td><td>IJCV 22</td><td></td><td>79.3/78.8</td><td>70.6/70.2</td><td>81.3/76.3</td><td>73.2/68.0</td><td>73.7/72.7</td><td>71.2/70.2</td><td>71.7/69.5</td></tr><tr><td>TransFusion [2]</td><td>CVPR 22</td><td></td><td>-/-</td><td>-/65.1</td><td>-/-</td><td>-/63.7</td><td>-/-</td><td>-/65.9</td><td>-/64.9</td></tr><tr><td>ConQueR [74]</td><td>CVPR 23</td><td></td><td>76.1/75.6</td><td>68.7/68.2</td><td>79.0/72.3</td><td>70.9/64.7</td><td>73.9/72.5</td><td>71.4/70.1</td><td>70.3/67.7</td></tr><tr><td>FocalFormer3D [7]</td><td>ICCV 23</td><td></td><td>-/-</td><td>68.1/67.6</td><td>-/-</td><td>72.7/66.8</td><td>-/-</td><td>73.7/72.6</td><td>71.5/69.0</td></tr><tr><td>HEDNet [68]</td><td>NeurIPS 23</td><td></td><td>81.1/80.6</td><td>73.2/72.7</td><td>84.4/80.0</td><td>76.8/72.6</td><td>78.7/77.7</td><td>75.8/74.9</td><td>75.3/73.4</td></tr><tr><td>SST_TS1 [15]</td><td>CVPR 22</td><td rowspan="5">Transformer</td><td>76.2/75.8</td><td>68.0/67.6</td><td>81.4/74.0</td><td>72.8/65.9</td><td>-/-</td><td>-/-</td><td>-/-</td></tr><tr><td>SWFormer [50]</td><td>ECCV 22</td><td>77.8/77.3</td><td>69.2/68.8</td><td>80.9/72.7</td><td>72.5/64.9</td><td>−/-</td><td>-/-</td><td>-/-</td></tr><tr><td>OcTr [70]</td><td>CVPR 23</td><td>78.1/77.6</td><td>69.8/69.3</td><td>80.8/74.4</td><td>72.5/66.5</td><td>72.6/71.5</td><td>69.9/68.9</td><td>70.7/68.2</td></tr><tr><td>DSVT-Pillar [57]</td><td>CVPR 23</td><td>79.3/78.8</td><td>70.9/70.5</td><td>82.8/77.0</td><td>75.2/69.8</td><td>76.4/75.4</td><td>73.6/72.7</td><td>73.2/71.0</td></tr><tr><td>DSVT-Voxel [57]</td><td>CVPR 23</td><td>79.7/79.3</td><td>71.4/71.0</td><td>83.7/78.9</td><td>76.1/71.5</td><td>77.5/76.5</td><td>74.6/73.7</td><td>74.0/72.1</td></tr><tr><td>LION-RetNet (Ours)</td><td>-</td><td rowspan="4">RNN</td><td>79.0/78.5</td><td>70.6/70.2</td><td>84.6/80.0</td><td>77.2/72.8</td><td>79.0/78.0</td><td>76.1/75.1</td><td>74.6/72.7</td></tr><tr><td>LION-RWKV (Ours)</td><td>-</td><td>79.7/79.3</td><td>71.3/71.0</td><td>84.6/80.0</td><td>77.1/72.7</td><td>78.7/77.7</td><td>75.8/74.8</td><td>74.7/72.8</td></tr><tr><td>LION-Mamba (Ours)</td><td>-</td><td>79.5/79.1</td><td>71.1/70.7</td><td>84.9/80.4</td><td>77.5/73.2</td><td>79.7/78.7</td><td>76.7/75.8</td><td>75.1/73.2</td></tr><tr><td>LION-Mamba-L (Ours)</td><td>-</td><td>80.3/79.9</td><td>72.0/71.6</td><td>85.8/81.4</td><td>78.5/74.3</td><td>80.1/79.0</td><td>77.2/76.2</td><td>75.9/74.0</td></tr></table>

Table 2: Performances on the nuScenes validation and test set. 'T.L.', 'C.V.', 'Ped.', 'M.T.', 'T.C.', and 'B.R.' are short for trailer, construction vehicle, pedestrian, motor, traffic cone, and barrier, respectively. All results are reported without any test-time augmentation and model ensembling.

<table><tr><td colspan="14">Performances on the validation set</td></tr><tr><td>Method</td><td>Present at</td><td>NDS</td><td>mAP</td><td>Car</td><td>Truck</td><td>Bus</td><td>T.L.</td><td>C.V.</td><td>Ped.</td><td>M.T.</td><td>Bike</td><td>T.C.</td><td>B.R.</td></tr><tr><td>CenterPoint [66]</td><td>CVPR 21</td><td>66.5</td><td>59.2</td><td>84.9</td><td>57.4</td><td>70.7</td><td>38.1</td><td>16.9</td><td>85.1</td><td>59.0</td><td>42.0</td><td>69.8</td><td>68.3</td></tr><tr><td>VoxelNeXt [9]</td><td>CVPR 23</td><td>66.7</td><td>60.5</td><td>83.9</td><td>55.5</td><td>70.5</td><td>38.1</td><td>21.1</td><td>84.6</td><td>62.8</td><td>50.0</td><td>69.4</td><td>69.4</td></tr><tr><td>Uni3DETR [58]</td><td>NeurIPS 23</td><td>68.5</td><td>61.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TransFusion-LiDAR [2]</td><td>CVPR 22</td><td>70.1</td><td>65.5</td><td>86.9</td><td>60.8</td><td>73.1</td><td>43.4</td><td>25.2</td><td>87.5</td><td>72.9</td><td>57.3</td><td>77.2</td><td>70.3</td></tr><tr><td>DSVT [57]</td><td>CVPR 23</td><td>71.1</td><td>66.4</td><td>87.4</td><td>62.6</td><td>75.9</td><td>42.1</td><td>25.3</td><td>88.2</td><td>74.8</td><td>58.7</td><td>77.9</td><td>71.0</td></tr><tr><td>HEDNet [68]</td><td>NeurIPS 23</td><td>71.4</td><td>66.7</td><td>87.7</td><td>60.6</td><td>77.8</td><td>50.7</td><td>28.9</td><td>87.1</td><td>74.3</td><td>56.8</td><td>76.3</td><td>66.9</td></tr><tr><td>LION-RetNet (Ours)</td><td>-</td><td>71.9</td><td>67.3</td><td>87.9</td><td>64.3</td><td>78.7</td><td>44.6</td><td>27.6</td><td>88.9</td><td>73.5</td><td>56.6</td><td>79.2</td><td>72.1</td></tr><tr><td>LION-RWKV (Ours)</td><td>-</td><td>71.7</td><td>66.8</td><td>88.1</td><td>59.0</td><td>77.6</td><td>46.6</td><td>28.0</td><td>89.7</td><td>74.3</td><td>56.2</td><td>80.1</td><td>68.3</td></tr><tr><td>LION-Mamba (Ours)</td><td>-</td><td>72.1</td><td>68.0</td><td>87.9</td><td>64.9</td><td>77.6</td><td>44.4</td><td>28.5</td><td>89.6</td><td>75.6</td><td>59.4</td><td>80.8</td><td>71.6</td></tr><tr><td colspan="14">Performances on the test set</td></tr><tr><td>TransFusion-LiDAR [2]</td><td>CVPR 22</td><td>70.2</td><td>65.5</td><td>86.2</td><td>56.7</td><td>66.3</td><td>58.8</td><td>28.2</td><td>86.1</td><td>68.3</td><td>44.2</td><td>82.0</td><td>78.2</td></tr><tr><td>DSVT [57]</td><td>CVPR 23</td><td>72.7</td><td>68.4</td><td>86.8</td><td>58.4</td><td>67.3</td><td>63.1</td><td>37.1</td><td>88.0</td><td>73.0</td><td>47.2</td><td>84.9</td><td>78.4</td></tr><tr><td>HEDNet [68]</td><td>NeurIPS 23</td><td>72.0</td><td>67.7</td><td>87.1</td><td>56.5</td><td>70.4</td><td>63.5</td><td>33.6</td><td>87.9</td><td>70.4</td><td>44.8</td><td>85.1</td><td>78.1</td></tr><tr><td>LION-Mamba (Ours)</td><td>-</td><td>73.9</td><td>69.8</td><td>87.2</td><td>61.1</td><td>68.9</td><td>65.0</td><td>36.3</td><td>90.0</td><td>74.0</td><td>49.2</td><td>87.3</td><td>79.5</td></tr></table>

<!-- Media -->

Compared with the transformer-based methods [15, 50, 57], our LION with different linear group RNN operators outperforms the previous state-of-the-art (SOTA) transformer-based 3D backbone DSVT-Voxel [57], illustrating the generalization of our proposed framework. To further scale up our LION, we present the performance of LION-Mamba-L by doubling the channel dimension of LION-Mamba. It can be observed that LION-Mamba-L significantly outperforms DSVT-Voxel with 1.9 mAPH/L2, leading to a new SOTA performance. The above promising results effectively demonstrate the superiority of our proposed LION.

Results on nuScenes. We also evaluate our LION on nuScenes validation and test set [4] further to verify the effectiveness of our LION. As shown in Table 2, on nuScenes validation set, our LION-RetNet, LION-RWKV, and LION-Mamba achieves 71.9, 71.7, and 72.1 NDS, respectively, which outperforms the previous advanced methods DSVT [57] and HEDNet [68]. Besides, our LION-Mamba even brings a new SOTA on nuScenes test benchmark, which beats the previous advanced method DSVT with 1.2 NDS and 1.4 mAP, effectively illustrating the superiority of our LION. Note that all results of our LION are conducted without any test-time augmentation and model ensembling.

<!-- Meanless: 8 -->


<!-- Media -->

Table 3: Comparison with prior methods on Argoverse V2 validation set. 'Vehicle', 'C-Barrel', 'MPC-Sign', 'A-Bus', 'C-Cone', 'V-Trailer', 'MBT', 'W-Device' and 'W-Rider' are short for regular vehicle, construction barrel, mobile pedestrian crossing sign, articulated bus, construction cone, vehicular trailer, message board trailer, wheeled device, and wheeled rider.

<table><tr><td>Method</td><td>mAP</td><td>Vehicle</td><td>5,655</td><td>Pedestrian</td><td>Stop Sign</td><td>Box Truck</td><td>Bollard</td><td>C-Barrel</td><td>Motorcyclist</td><td>MPC-Sign</td><td>Motorcycle</td><td>Bicycle</td><td>A-Bus</td><td>School Bus</td><td>Truck Cab</td><td>C-Cone</td><td>V-Trailer</td><td>Sign</td><td>Large Vehicle</td><td>Stroller</td><td>Bicyclist</td><td>Truck</td><td>MBT</td><td>20,000</td><td>Wheelchair</td><td>W-Device</td><td>W-Rider</td></tr><tr><td>CenterPoint [66]</td><td>22.0</td><td>67.6</td><td>38.9</td><td>46.5</td><td>16.9</td><td>37.4</td><td>40.1</td><td>32.2</td><td>28.6</td><td>27.4</td><td>33.4</td><td>24.5</td><td>8.7</td><td>25.8</td><td>22.6</td><td>29.5</td><td>22.4</td><td>6.3</td><td>3.9</td><td>0.5</td><td>20.1</td><td>22.1</td><td>0.0</td><td>3.9</td><td>0.5</td><td>10.9</td><td>4.2</td></tr><tr><td>HEDNet [68]</td><td>37.1</td><td>78.2</td><td>47.7</td><td>67.6</td><td>46.4</td><td>45.9</td><td>56.9</td><td>67.0</td><td>48.7</td><td>46.5</td><td>58.2</td><td>47.5</td><td>23.3</td><td>40.9</td><td>27.5</td><td>46.8</td><td>27.9</td><td>20.6</td><td>6.9</td><td>27.2</td><td>38.7</td><td>21.6</td><td>0.0</td><td>30.7</td><td>9.5</td><td>28.5</td><td>8.7</td></tr><tr><td>VoxelNeXt [9]</td><td>30.7</td><td>72.7</td><td>38.8</td><td>63.2</td><td>40.2</td><td>40.1</td><td>53.9</td><td>64.9</td><td>44.7</td><td>39.4</td><td>42.4</td><td>40.6</td><td>20.1</td><td>25.2</td><td>19.9</td><td>44.9</td><td>20.9</td><td>14.9</td><td>6.8</td><td>15.7</td><td>32.4</td><td>16.9</td><td>0.0</td><td>14.4</td><td>0.1</td><td>17.4</td><td>6.6</td></tr><tr><td>FSDv1 [17]</td><td>28.2</td><td>68.1</td><td>40.9</td><td>59.0</td><td>29.0</td><td>38.5</td><td>41.8</td><td>42.6</td><td>39.7</td><td>26.2</td><td>49.0</td><td>38.6</td><td>20.4</td><td>30.5</td><td>14.8</td><td>41.2</td><td>26.9</td><td>11.9</td><td>5.9</td><td>13.8</td><td>33.4</td><td>21.1</td><td>0.0</td><td>9.5</td><td>7.1</td><td>14.0</td><td>9.2</td></tr><tr><td>FSDv2 [18]</td><td>37.6</td><td>77.0</td><td>47.6</td><td>70.5</td><td>43.6</td><td>41.5</td><td>53.9</td><td>58.5</td><td>56.8</td><td>39.0</td><td>60.7</td><td>49.4</td><td>28.4</td><td>41.9</td><td>30.2</td><td>44.9</td><td>33.4</td><td>16.6</td><td>7.3</td><td>32.5</td><td>45.9</td><td>24.0</td><td>1.0</td><td>12.6</td><td>17.1</td><td>26.3</td><td>17.2</td></tr><tr><td>SAFDNet [67]</td><td>39.7</td><td>78.5</td><td>49.4</td><td>70.7</td><td>51.5</td><td>44.7</td><td>65.7</td><td>72.3</td><td>54.3</td><td>49.7</td><td>60.8</td><td>50.0</td><td>31.3</td><td>44.9</td><td>24.7</td><td>55.4</td><td>31.4</td><td>22.1</td><td>7.1</td><td>31.1</td><td>42.7</td><td>23.6</td><td>0.0</td><td>26.1</td><td>1.4</td><td>30.2</td><td>11.5</td></tr><tr><td>LION-RetNet</td><td>40.7</td><td>74.7</td><td>41.0</td><td>72.7</td><td>47.5</td><td>44.2</td><td>66.9</td><td>77.0</td><td>57.1</td><td>48.3</td><td>63.7</td><td>55.1</td><td>27.0</td><td>42.5</td><td>25.2</td><td>57.9</td><td>29.7</td><td>22.0</td><td>6.9</td><td>39.3</td><td>47.3</td><td>19.9</td><td>0.0</td><td>28.8</td><td>12.8</td><td>37.7</td><td>12.6</td></tr><tr><td>LION-RWKV</td><td>41.1</td><td>76.3</td><td>44.6</td><td>74.0</td><td>52.1</td><td>46.0</td><td>68.1</td><td>75.8</td><td>55.8</td><td>49.4</td><td>62.8</td><td>55.3</td><td>27.1</td><td>42.9</td><td>25.9</td><td>60.1</td><td>30.9</td><td>22.2</td><td>9.3</td><td>36.5</td><td>55.3</td><td>23.2</td><td>0.0</td><td>27.8</td><td>7.1</td><td>37.6</td><td>11.4</td></tr><tr><td>LION-Mamba</td><td>41.5</td><td>75.1</td><td>43.6</td><td>73.9</td><td>53.9</td><td>45.1</td><td>66.4</td><td>74.7</td><td>61.3</td><td>48.7</td><td>65.1</td><td>56.2</td><td>21.7</td><td>42.7</td><td>25.3</td><td>58.4</td><td>28.9</td><td>23.6</td><td>8.3</td><td>49.5</td><td>47.3</td><td>19.0</td><td>0.0</td><td>31.4</td><td>8.7</td><td>37.6</td><td>11.8</td></tr></table>

Table 4: Comparison with previous methods on ONCE validation set. We use the center head of CenterPoint for a fair comparison.

<table><tr><td rowspan="2">Method</td><td colspan="4">Vehicle</td><td colspan="4">Pedestrian</td><td colspan="4">Cyclist</td><td rowspan="2">mAP</td></tr><tr><td>overall</td><td>0-30m</td><td>30-50m</td><td>50m-inf</td><td>overall</td><td>0-30m</td><td>30-50m</td><td>50m-inf</td><td>overall</td><td>0-30m</td><td>30-50m</td><td>50m-inf</td></tr><tr><td>PointRCNN [47]</td><td>52.1</td><td>74.5</td><td>40.9</td><td>16.8</td><td>4.3</td><td>6.2</td><td>2.4</td><td>0.9</td><td>29.8</td><td>46.0</td><td>20.9</td><td>5.5</td><td>28.7</td></tr><tr><td>PointPillars [26]</td><td>68.6</td><td>80.9</td><td>62.1</td><td>47.0</td><td>17.6</td><td>19.7</td><td>15.2</td><td>10.2</td><td>46.8</td><td>58.3</td><td>40.3</td><td>25.9</td><td>44.3</td></tr><tr><td>SECOND [60]</td><td>71.2</td><td>84.0</td><td>63.0</td><td>47.3</td><td>26.4</td><td>29.3</td><td>24.1</td><td>18.1</td><td>58.0</td><td>70.0</td><td>52.4</td><td>34.6</td><td>51.9</td></tr><tr><td>PV-RCNN [45]</td><td>77.8</td><td>89.4</td><td>72.6</td><td>58.6</td><td>23.5</td><td>25.6</td><td>22.8</td><td>17.3</td><td>59.4</td><td>71.7</td><td>52.6</td><td>36.2</td><td>53.6</td></tr><tr><td>CenterPoint [66]</td><td>66.8</td><td>80.1</td><td>59.6</td><td>43.4</td><td>49.9</td><td>56.2</td><td>42.6</td><td>26.3</td><td>63.5</td><td>74.3</td><td>57.9</td><td>41.5</td><td>60.1</td></tr><tr><td>PointPainting [54]</td><td>66.2</td><td>80.3</td><td>59.8</td><td>42.3</td><td>44.8</td><td>52.6</td><td>36.6</td><td>22.5</td><td>62.3</td><td>73.6</td><td>57.2</td><td>40.4</td><td>57.8</td></tr><tr><td>LION-RetNet</td><td>78.1</td><td>88.7</td><td>72.4</td><td>58.5</td><td>52.4</td><td>60.5</td><td>43.6</td><td>26.3</td><td>68.3</td><td>79.4</td><td>62.9</td><td>46.1</td><td>66.3</td></tr><tr><td>LION-RWKV</td><td>78.3</td><td>89.2</td><td>72.6</td><td>56.7</td><td>50.6</td><td>60.0</td><td>40.4</td><td>24.2</td><td>68.4</td><td>79.4</td><td>63.2</td><td>45.7</td><td>65.8</td></tr><tr><td>LION-Mamba</td><td>78.2</td><td>89.1</td><td>72.6</td><td>57.5</td><td>53.2</td><td>62.4</td><td>44.0</td><td>24.5</td><td>68.5</td><td>79.2</td><td>63.2</td><td>47.1</td><td>66.6</td></tr></table>

Table 5: Ablation study for each component in LION. Here, the large group size means that we set it as (4096, 2048, 1024, 512) for four blocks (also refer to the section of our implementation details), otherwise, we set a small group size as (256, 256, 256, 256).

<table><tr><td rowspan="2">Large Group Size</td><td rowspan="2">3D Spatial Feature Descriptor</td><td rowspan="2">Voxel Generation</td><td colspan="3">3D AP/APH (L2)</td><td rowspan="2">mAP/mAPH (L2)</td></tr><tr><td>Vehicle</td><td>Pedestrian</td><td>Cyclist</td></tr><tr><td>-</td><td>-</td><td>-</td><td>65.6/65.2</td><td>72.3/65.0</td><td>68.3/67.2</td><td>68.8/65.8</td></tr><tr><td>✓</td><td>-</td><td>-</td><td>66.2/65.7</td><td>73.7/67.2</td><td>68.7/67.6</td><td>69.5/66.9</td></tr><tr><td>✓</td><td>✓</td><td>-</td><td>66.5/66.1</td><td>74.8/69.6</td><td>70.9/70.0</td><td>70.8/68.6</td></tr><tr><td>✓</td><td>-</td><td>✓</td><td>66.4/66.0</td><td>73.5/67.4</td><td>70.4/69.3</td><td>70.1/67.6</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>67.0/66.6</td><td>75.4/70.2</td><td>71.9/71.0</td><td>71.4/69.3</td></tr></table>

<!-- Media -->

Results on Argoverse V2. To further verify the effectiveness of our LION on the long-range perception, we evaluate the experiments on Argoverse V2 validation set. For a fair comparison, we adopt the same detection head [66] with VoxelNext [9] and SAFDNet [67] for long-range perception. As shown in Table 3, our LION-RetNet, LION-RWKV, and LION-Mamba achieve the detection performance with ${40.7}\mathrm{{mAP}},{41.1}\mathrm{{mAP}}$ and ${41.5}\mathrm{{mAP}}$ ,all three of which have outperformed the previous SOTA method SAFDNet [67], leading to new SOTA results. These superior results clearly illustrate the effectiveness of our LION.

Results on ONCE. We also evaluated our LION on ONCE validation set to further verify the effectiveness of our LION. As shown in Table 4, our LION-RetNet, LION-RWKV, and LION-Mamba produces advanced detection performance with ${66.3}\mathrm{{mAP}},{65.8}\mathrm{{mAP}}$ ,and ${66.6}\mathrm{{mAP}}$ ,respectively. It is worth mentioning that our LION-Mamba outperforms the previous SOTA method CenterPoint [66] with ${6.5}\mathrm{{mAP}}$ ,leading to a new SOTA result. These experiments illustrate the superiority of our LION.

### 4.4 Ablation Study

In this section, we conduct ablation studies of LION on the WOD validation set with 20% training data. If not specified, we adopt LION-Mamba as our default model and train our model with 12 epochs in the following ablation studies. For more experiments, please refer to our appendix.

Ablation Study of LION. To illustrate the effectiveness of our proposed LION, we conduct the ablation study for each component, including the design of large group size, 3D spatial feature descriptor, and voxel generation in Table 5. Here, our baseline is proposed LION that removes the design of large group size, 3D spatial feature descriptor, and voxel generation. In Table 5, we observe that the design of large group size even brings 1.1 mAPH/L2 performance improvement, which illustrates the benefits of performing long-range feature interaction with the help of linear RNN. Then, we integrate the 3D spatial feature descriptor, which further produces an obvious performance improvement with ${1.7}\mathrm{{mAPH}}/\mathrm{L}2$ . This demonstrates the superiority of the $3\mathrm{D}$ spatial feature descriptor in compensating for the lack of capturing spatial information of linear RNNs. Furthermore, we notice that the 3D spatial feature descriptor is very helpful to small objects (e.g., Pedestrians) thanks to its capability of extracting the local information of 3D objects. To address the challenge of feature representation in highly sparse point clouds, we adopt voxel generation to enhance the features of foregrounds, which brings a promising gain of 0.7 mAPH/L2 (67.6 vs. 66.9). By combining all components, our LION achieves a superior performance of 69.3 mAPH/L2, which outperforms the baseline of ${3.5}\mathrm{{mAPH}}/\mathrm{L}2$ .

<!-- Meanless: 9 -->


<!-- Media -->

Table 6: Ablation study for 3D Spatial Feature Descriptor (3D SFD) in LION.

<table><tr><td rowspan="2">Methods</td><td colspan="3">3D AP/APH (L2)</td><td rowspan="2">mAP/mAPH (L2)</td></tr><tr><td>Vehicle</td><td>Pedestrian</td><td>Cyclist</td></tr><tr><td>Baseline</td><td>66.4/66.0</td><td>73.5/67.4</td><td>70.4/69.3</td><td>70.1/67.6</td></tr><tr><td>MLP</td><td>66.6/66.2</td><td>74.1/68.1</td><td>70.0/69.0</td><td>70.2/67.7</td></tr><tr><td>Linear Group RNN</td><td>66.4/66.0</td><td>74.0/68.2</td><td>70.5/69.5</td><td>70.3/67.9</td></tr><tr><td>3D SFD (Ours)</td><td>67.0/66.6</td><td>75.4/70.2</td><td>71.9/71.0</td><td>71.4/69.3</td></tr></table>

Table 7: Ablation study for voxel generation in LION. "Baseline" indicates no voxel generation. "Zero Feats" and "K-NN Feats" indicate initializing features to all zeros and K-NN features, respectively. "Auto-Regressive" uses the LION block based on linear group RNN for its auto-regressive property. "Sparse-Conv" maintains the same structure as the LION block but replaces the linear group RNN with 3D sub-manifold convolution.

<table><tr><td rowspan="2">Index</td><td rowspan="2">Methods</td><td colspan="3">3D AP/APH (L2)</td><td rowspan="2">mAP/mAPH (L2)</td></tr><tr><td>Vehicle</td><td>Pedestrian</td><td>Cyclist</td></tr><tr><td>I</td><td>Baseline</td><td>66.5/66.1</td><td>74.8/69.6</td><td>70.9/70.0</td><td>70.8/68.6</td></tr><tr><td>II</td><td>Zero Feats + Sparse-Conv</td><td>64.6/64.2</td><td>72.8/67.4</td><td>69.3/68.3</td><td>68.9/66.6</td></tr><tr><td>III</td><td>K-NN Feats + Auto-Regressive</td><td>66.5/66.1</td><td>74.0/68.7</td><td>71.1/70.1</td><td>70.5/68.3</td></tr><tr><td>IV</td><td>Zero Feats + Auto-Regressive (Ours)</td><td>67.0/66.6</td><td>75.4/70.2</td><td>71.9/71.0</td><td>71.4/69.3</td></tr></table>

<!-- Media -->

Superiority of 3D Spatial Feature Descriptor. To further verify the necessity of 3D spatial feature descriptor, we provide the comparison with two available manners including the MLP and linear RNN to replace our descriptor in Table 6. Here, we set our LION without 3D spatial feature descriptor as the baseline in this part. We observe that MLP even does not bring promising performance improvement in terms of mAPH/L2 since MLP lacks the ability to capture local 3D spatial information. Furthermore, considering the limited receptive field of MLP, we adopt a linear group RNN operator to replace MLP. We find that there is only slight performance improvement with ${0.3}\mathrm{{mAPH}}/\mathrm{L}2$ ,which indicates that the linear group RNN might not be good at modeling local spatial relationships although it has the strong capability to establish long-range relationships. In contrast, our 3D spatial feature descriptor brings obvious performance improvement, which boosts the baseline of 1.7 mAPH/L2. This effectively illustrates the superiority of the 3D spatial feature descriptor in compensating for the lack of local 3D spatial-aware modeling in the linear group RNN.

Effectiveness of Voxel Generation. Voxel generation is applied to enhance the feature representation of objects in highly sparse point clouds for accurate 3D object detection. Therefore, to explore the effectiveness of our proposed voxel generation, we present the comparison with several available manners in Table 7. First, we compare our results of IV with II by only replacing the operator of linear group RNN in LION block with 3D sub-manifold convolution to generate the diffused features. We find that the manner of IV (69.3 vs. 66.6) significantly outperforms the performance of II in terms of mAPH/L2. This benefits from the linear group RNN's ability to model long-range feature interactions, generating a more reliable feature representation through its auto-regressive capacity, demonstrating the superiority of voxel generation with the linear group RNN. To further illustrate that the effectiveness of voxel generation is from its auto-regressive property of LION block rather than a strong feature extractor, we initialize the diffused features of the foreground voxels by K-NN operation (III) instead of the manner of all-zeros features (IV) and then feed them to the same following LION block for voxel generation. In Table 7, we find that the manner of III is inferior to IV by 1.0 mAPH/L2. This clearly illustrates that our voxel generation is benefiting from its auto-regressive property of LION block. Finally, compared with the baseline (I), our voxel generation (IV) can obtain a promising performance improvement, which verifies its effectiveness.

<!-- Meanless: 10 -->


## 5 Conclusion

In this paper, we have presented a simple and effective window-based framework termed LION, which can capture the long-range relationship by adopting linear RNN for large groups. Specifically, LION incorporates a proposed LION block to unlock the great potential of linear RNNs in modeling a long-range relationship and a voxel generation strategy to obtain more discriminative feature representation in sparse point clouds. Extensive ablation studies demonstrate the effectiveness of our proposed components. Additionally, the generalization of our LION is verified by performing different linear group RNN operators. Benefiting from our well-designed framework and the proposed superior components, our LION-Mamba achieves state-of-the-art performance on the challenging Waymo and nuScenes datasets.

Limitations. Although our LION based on the linear group RNN can perform long-range feature interaction with linear complexity, the corresponding running speed still needs further improvement since linear RNNs are not as efficient as transformers in parallel computing.

## References

[1] Benedikt Alkin, Maximilian Beck, Korbinian Pöppel, Sepp Hochreiter, and Johannes Brandstetter. Vision-Istm: xlstm as generic vision backbone. arXiv:2406.04303, 2024. 3

[2] Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, and Chiew-Lan Tai. Transfusion: Robust lidar-camera fusion for 3d object detection with transformers. In CVPR, 2022. 7, 8

[3] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long short-term memory. arXiv:2405.04517, 2024. 3, 15

[4] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 2, 3, 7, 8

[5] Chen Chen, Zhe Chen, Jing Zhang, and Dacheng Tao. Sasa: Semantics-augmented set abstraction for point-based 3d object detection. In AAAI, 2022. 3

[6] Yilun Chen, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Fast point r-cnn. In ICCV, 2019. 3

[7] Yilun Chen, Zhiding Yu, Yukang Chen, Shiyi Lan, Anima Anandkumar, Jiaya Jia, and Jose M Alvarez. Focalformer3d: Focusing on hard instance for 3d object detection. In ICCV, 2023. 8

[8] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Largekernel34: Scaling up kernels in 3d sparse cnns. In CVPR, 2023. 3

[9] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. In CVPR, 2023. 7, 8, 9

[10] Bowen Cheng, Lu Sheng, Shaoshuai Shi, Ming Yang, and Dong Xu. Back-tracing representative points for voting-based 3d object detection in point clouds. In CVPR, 2021. 3

[11] Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, and Caglar Gulcehre. Griffin: Mixing gated linear recurrences with local attention for efficient language models. arXiv:2402.19427, 2024. 3

[12] Jiajun Deng, Shaoshuai Shi, Peiwei Li, Wengang Zhou, Yanyong Zhang, and Houqiang Li. Voxel r-cnn: Towards high performance voxel-based 3d object detection. In AAAI, 2021. 3

[13] Shaocong Dong, Lihe Ding, Haiyang Wang, Tingfa Xu, Xinli Xu, Jie Wang, Ziyang Bian, Ying Wang, and Jianan Li. Mssvt: Mixed-scale sparse voxel transformer for 3d object detection on point clouds. In NeurIPS, 2022. 3

[14] Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, and Wenhai Wang. Vision-rwkv: Efficient and scalable visual perception with rwkv-like architectures. arXiv:2403.02308, 2024. 3

<!-- Meanless: 11 -->


[15] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Embracing single stride 3d object detector with sparse transformer. In CVPR, 2022. 1, 3, 4, 8, 17

[16] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Embracing single stride 3d object detector with sparse transformer. In CVPR, 2022. 3

[17] Lue Fan, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Fully sparse 3d object detection. In NeurIPS, 2022.2, 5, 8, 9

[18] Lue Fan, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Fsd v2: Improving fully sparse 3d object detection with virtual voxels. arXiv:2308.03755, 2023. 9

[19] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. IJRS, 2013. 7

[20] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In CVPR, 2012. 7, 15

[21] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In ICASSP, 2013. 7

[22] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752, 2023. 2, 3, 4, 7, 15, 16

[23] Tianrui Guan, Jun Wang, Shiyi Lan, Rohan Chandra, Zuxuan Wu, Larry Davis, and Dinesh Manocha. M3detr: Multi-representation, multi-scale, mutual-relation 3d object detection with transformers. In WACV, 2022. 3

[24] Chenhang He, Hui Zeng, Jianqiang Huang, Xian-Sheng Hua, and Lei Zhang. Structure aware single-stage 3d object detection from point cloud. In CVPR, 2020. 3

[25] Yihan Hu, Zhuangzhuang Ding, Runzhou Ge, Wenxin Shao, Li Huang, Kun Li, and Qiang Liu. Afdetv2: Rethinking the necessity of the second stage for object detection from point clouds. In AAAI, 2022. 8

[26] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In CVPR, 2019. 8, 9, 15

[27] Jinyu Li, Chenxu Luo, and Xiaodong Yang. Pillarnext: Rethinking network designs for 3d object detection in lidar point clouds. In CVPR, 2023. 8

[28] Dingkang Liang, Xin Zhou, Xinyu Wang, Xingkui Zhu, Wei Xu, Zhikang Zou, Xiaoqing Ye, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. arXiv:2402.10739, 2024. 3

[29] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, and Yunfan Liu. Vmamba: Visual state space model. arXiv:2401.10166, 2024. 5

[30] Ze Liu, Zheng Zhang, Yue Cao, Han Hu, and Xin Tong. Group-free 3d object detection via transformers. In ICCV, 2021. 3

[31] Zhe Liu, Xin Zhao, Tengteng Huang, Ruolan Hu, Yu Zhou, and Xiang Bai. Tanet: Robust 3d object detection from point clouds with triple attention. In AAAI, 2020. 3, 15

[32] Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, and Song Han. Flatformer: Flattened window attention for efficient point cloud transformer. In CVPR, 2023. 3, 4, 17

[33] Tao Lu, Xiang Ding, Haisong Liu, Gangshan Wu, and Limin Wang. Link: Linear kernel for lidar-based 3d perception. In CVPR, 2023. 3

[34] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, et al. One million scenes for autonomous driving: Once dataset. arXiv:2106.11037, 2021. 2, 3, 7

[35] Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In ICML, 2023. 3

[36] Xuran Pan, Zhuofan Xia, Shiji Song, Li Erran Li, and Gao Huang. 3d object detection with pointformer. In CVPR, 2021. 3

<!-- Meanless: 12 -->


[37] Bo Peng, Eric Alcaide, Quentin Gregory Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Nguyen Chung, Leon Derczynski, et al. Rwkv: Reinventing rnns for the transformer era. In EMNLP, 2023. 2, 3, 4, 7, 15

[38] Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Xingjian Du, Teddy Ferdinan, Haowen Hou, Przemysław Kazienko, Kranthi Kiran GV, Jan Kocón, Bartlomiej Koptyra, Satyapriya Krishna, Ronald McClelland Jr. au2, Niklas Muennighoff, Fares Obeid, Atsushi Saito, Guangyu Song, Haoqin Tu, Stanistaw Woźniak, Ruichong Zhang, Bingchen Zhao, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui-Jie Zhu. Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence. arXiv:2404.05892, 2024. 3, 16

[39] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In ICCV, 2019. 3

[40] Charles R Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J Guibas. Frustum pointnets for 3d object detection from rgb-d data. In CVPR, 2018. 3, 15

[41] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 3

[42] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017. 3

[43] Zhen Qin, Songlin Yang, and Yiran Zhong. Hierarchically gated recurrent neural network for sequence modeling. In NeurIPS, 2023. 3

[44] Guangsheng Shi, Ruifeng Li, and Chao Ma. Pillarnet: High-performance pillar-based 3d object detection. In ECCV, 2022. 8

[45] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In CVPR, 2020. 3, 8, 9

[46] Shaoshuai Shi, Li Jiang, Jiajun Deng, Zhe Wang, Chaoxu Guo, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn++: Point-voxel feature set abstraction with local vector representation for $3\mathrm{\;d}$ object detection. IJCV, 2021. 3, 8

[47] Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. Pointrenn: 3d object proposal generation and detection from point cloud. In CVPR, 2019. 3, 9, 15

[48] Shaoshuai Shi, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. From points to parts: 3d object detection from point cloud with part-aware and part-aggregation network. IEEE TPAMI, 2020. 3

[49] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR, 2020. 2, 3, 6, 7, 15, 17

[50] Pei Sun, Mingxing Tan, Weiyue Wang, Chenxi Liu, Fei Xia, Zhaoqi Leng, and Drago Anguelov. Swformer: Sparse window transformer for 3d object detection in point clouds. In ECCV, 2022. 1, 2, 3, 5, 6, 8

[51] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv:2407.04620, 2024. 3, 15

[52] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv:2307.08621, 2023. 3,7,15,16

[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1, 3

[54] Sourabh Vora, Alex H Lang, Bassam Helou, and Oscar Beijbom. Pointpainting: Sequential fusion for 3d object detection. In CVPR, 2020. 9

[55] Chunwei Wang, Chao Ma, Ming Zhu, and Xiaokang Yang. Pointaugmenting: Cross-modal augmentation for 3d object detection. In CVPR, 2021. 7

[56] Haiyang Wang, Lihe Ding, Shaocong Dong, Shaoshuai Shi, Aoxue Li, Jianan Li, Zhenguo Li, and Liwei Wang. Cagroup3d: Class-aware grouping for 3d object detection on point clouds. In NeurIPS, 2022. 3

<!-- Meanless: 13 -->


[57] Haiyang Wang, Chen Shi, Shaoshuai Shi, Meng Lei, Sen Wang, Di He, Bernt Schiele, and Liwei Wang. Dsvt. Dynamic sparse voxel transformer with rotated sets. In CVPR, 2023. 1, 2, 3, 4, 7, 8, 15, 17

[58] Zhenyu Wang, Ya-Li Li, Xi Chen, Hengshuang Zhao, and Shengjin Wang. Uni3detr: Unified 3d detection transformer. In NeurIPS, 2024. 8

[59] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. In NeurIPS, 2023. 2, 3, 7

[60] Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 2018. 3, 4, 7, 8, 9, 15

[61] Honghui Yang, Wenxiao Wang, Minghao Chen, Binbin Lin, Tong He, Hua Chen, Xiaofei He, and Wanli Ouyang. Pvt-ssd: Single-stage 3d object detector with point-voxel transformer. In CVPR, 2023. 3

[62] Jinrong Yang, Lin Song, Songtao Liu, Weixin Mao, Zeming Li, Xiaoping Li, Hongbin Sun, Jian Sun, and Nanning Zheng. Dbq-ssd: Dynamic ball query for efficient 3d object detection. In ICLR, 2022. 3

[63] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv:2312.06635, 2023. 3

[64] Zetong Yang, Yanan Sun, Shu Liu, and Jiaya Jia. 3dssd: Point-based 3d single stage object detector. In CVPR, 2020. 3

[65] Zetong Yang, Yanan Sun, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Std: Sparse-to-dense 3d object detector for point cloud. In CVPR, 2019. 3

[66] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Center-based 3d object detection and tracking. In CVPR, 2021. 3, 4, 8, 9, 17

[67] Gang Zhang, Junnan Chen, Guohuan Gao, Jianmin Li, Si Liu, and Xiaolin Hu. Safdnet: A simple and effective network for fully sparse 3d object detection. In CVPR, 2024. 2, 3, 5, 7, 9

[68] Gang Zhang, Chen Junnan, Guohuan Gao, Jianmin Li, and Xiaolin Hu. HEDNet: A hierarchical encoder-decoder network for 3d object detection in point clouds. In NeurIPS, 2023. 8, 9

[69] Yifan Zhang, Qingyong Hu, Guoquan Xu, Yanxin Ma, Jianwei Wan, and Yulan Guo. Not all points are equal: Learning highly efficient point-based detectors for 3d lidar point clouds. In CVPR, 2022. 3

[70] Chao Zhou, Yanan Zhang, Jiaxin Chen, and Di Huang. Octr: Octree-based transformer for 3d object detection. In CVPR, 2023. 8

[71] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In CVPR, 2018. 15

[72] Zixiang Zhou, Xiangchen Zhao, Yu Wang, Panqu Wang, and Hassan Foroosh. Centerformer: Center-based transformer for 3d object detection. In ECCV, 2022. 8

[73] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu. Class-balanced grouping and sampling for point cloud 3d object detection. arXiv:1908.09492, 2019. 7

[74] Benjin Zhu, Zhe Wang, Shaoshuai Shi, Hang Xu, Lanqing Hong, and Hongsheng Li. Conquer: Query contrast voxel-detr for 3d object detection. In CVPR, 2023. 8

<!-- Meanless: 14 -->


## A Appendix

The appendix is organized as follows. First, in section A.1, we provide more types of linear RNN operators (e.g., RetNet, RWKV, Mamba, xLSTM, and TTT) based on our LION framework for 3D detection on a small but popular dataset KITTI for a quick experience. Second, we present extra experiments on the WOD [49] validation set in section A.2, including the placement of the 3D spatial feature descriptor, the impact of different window sizes and different group sizes in inference, and the ratio $r$ in voxel generation. Third,we provide the comparison of computation cost and parameter size in section A.3, and detailed information of LION structure in section A.4. Forth, in section A.5, we visualize the feature maps of different LION blocks to illustrate the rationality of distinguishing foreground voxels based on feature response. Finally, we provide the comparison of qualitative results with DSVT [57] and qualitative results of LION to demonstrate the superiority of our LION in section A.6 and section A.7. Besides, we discuss the broader impacts in section A.8.

### A.1 Experiments on KITTI dataset

<!-- Media -->

Table 8: Effectiveness on the KITTI validation set for Car, Pedestrian, and Cyclist. * represents our reproduced results by keeping the same configures except for their 3D backbones for a fair comparison. Our LION supports different representative linear RNN operators (TTT, xLSTM, RetNet, RWKV, and Mamba). mAP is calculated by all categories and all difficulties with recall 11.

<table><tr><td rowspan="2">Method</td><td colspan="3">Car</td><td colspan="3">Pedestrian</td><td colspan="3">Cyclist</td><td rowspan="2">mAP</td></tr><tr><td>Easy</td><td>Moderate</td><td>Hard</td><td>Easy</td><td>Moderate</td><td>Hard</td><td>Easy</td><td>Moderate</td><td>Hard</td></tr><tr><td>VoxelNet [71]</td><td>77.5</td><td>65.1</td><td>57.7</td><td>39.5</td><td>33.7</td><td>31.5</td><td>61.2</td><td>48.4</td><td>44.4</td><td>51.0</td></tr><tr><td>SECOND [60]</td><td>83.1</td><td>73.7</td><td>66.2</td><td>51.1</td><td>42.6</td><td>37.3</td><td>70.5</td><td>53.9</td><td>46.9</td><td>58.4</td></tr><tr><td>PointPillars [26]</td><td>79.1</td><td>75.0</td><td>68.3</td><td>52.1</td><td>43.5</td><td>41.5</td><td>75.8</td><td>59.1</td><td>52.9</td><td>60.8</td></tr><tr><td>PointRCNN [47]</td><td>85.9</td><td>75.8</td><td>68.3</td><td>49.4</td><td>41.8</td><td>38.6</td><td>73.9</td><td>59.6</td><td>53.6</td><td>60.8</td></tr><tr><td>TANet [31]</td><td>83.8</td><td>75.4</td><td>67.7</td><td>54.9</td><td>46.7</td><td>42.4</td><td>73.8</td><td>59.9</td><td>53.5</td><td>62.0</td></tr><tr><td>DSVT-Pillar* [57]</td><td>87.3</td><td>77.4</td><td>76.2</td><td>61.4</td><td>56.8</td><td>51.8</td><td>82.3</td><td>67.1</td><td>63.7</td><td>69.3</td></tr><tr><td>DSVT-Voxel* [57]</td><td>87.8</td><td>77.8</td><td>76.8</td><td>66.1</td><td>59.7</td><td>55.2</td><td>83.5</td><td>66.7</td><td>63.2</td><td>70.8</td></tr><tr><td>LION-TTT</td><td>87.9</td><td>78.0</td><td>76.7</td><td>63.4</td><td>58.6</td><td>53.7</td><td>84.0</td><td>69.6</td><td>64.5</td><td>70.7</td></tr><tr><td>LION-xLSTM</td><td>87.7</td><td>77.9</td><td>76.8</td><td>66.6</td><td>59.3</td><td>54.0</td><td>82.4</td><td>67.4</td><td>63.4</td><td>70.6</td></tr><tr><td>LION-RetNet</td><td>88.0</td><td>77.9</td><td>76.7</td><td>67.4</td><td>60.2</td><td>55.8</td><td>83.6</td><td>69.6</td><td>64.6</td><td>71.5</td></tr><tr><td>LION-Mamba</td><td>88.6</td><td>78.3</td><td>77.2</td><td>67.2</td><td>60.2</td><td>55.6</td><td>83.0</td><td>68.6</td><td>63.9</td><td>71.4</td></tr><tr><td>LION-RWKV</td><td>88.5</td><td>78.3</td><td>77.1</td><td>68.9</td><td>62.2</td><td>58.1</td><td>89.6</td><td>71.2</td><td>66.9</td><td>73.4</td></tr></table>

<!-- Media -->

KITTI Dataset. KITTI [20] is a popular benchmark dataset for autonomous driving, which consists of 7481 training frames and 7518 test frames for 3D object detection. We follow the dataset splitting protocol in [40] and further split the 7481 training frames into 3712 frames for training set and 3769 frames for validation set. For the 3D detection task, KITTI dataset mainly detects Car, Pedestrian, and Cyclist for three difficulty levels, i.e., Easy, Moderate, and Hard. And the mean Average Precision (mAP) using 11 recall positions is adopted as the evaluation metric.

Results on KITTI. We conduct experiments on the KITTI validation set to illustrate the generalization of LION for different linear RNN operators. We select some representative linear RNN operators (TTT [51], xLSTM [3], RetNet [52], Mamba [22], and RWKV [37]) for LION. We adopt the same training parameters (i.e., number of epochs, learning rate, optimizer) with SECOND [60]. Besides, we use the same BEV backbone and the detection head with SECOND [60]. For a fair comparison, we keep the same configure of DSVT [57] and all our LION methods except 3D backbones. As shown in Table 8, LION-RetNet, LION-Mamba, and LION-RWKV outperforms DSVT-Voxel by 0.7 mAP, 0.6 mAP, and 2.6 mAP. These experiments demonstrate the generalization and effectiveness of our linear RNN-based framework LION.

### A.2 Extra Experiments

The Placement of 3D Spatial Feature Descriptor. We conduct experiments about the placement of the 3D spatial feature descriptor, as shown in Table 9. We regard the manner that does not adopt the 3D SFD of our LION as the baseline. Here, we provide two available manners: Placement 1 and Placement 2. For Placement 1, we place the 3D SFD after voxel merging. For Placement 2, we place 3D SFD before the voxel merging. Compared to the baseline, Placement 1 brings 1.0 mAPH/L2 improvement, which demonstrates the effectiveness of 3D SFD in compensating for the lack of local 3D spatial-aware modeling in linear RNNs. Moreover, Placement 2 further brings 0.7 mAPH/L2 improvement over Placement 1, which demonstrates the effectiveness of 3D SFD for reducing spatial information loss in the process of voxel merging.

<!-- Meanless: 15 -->


<!-- Media -->

Table 9: The Placement of 3D Spatial Feature Descriptor.

<table><tr><td rowspan="2">Methods</td><td colspan="3">3D AP/APH (L2)</td><td rowspan="2">mAP/mAPH (L2)</td></tr><tr><td>Vehicle</td><td>Pedestrian</td><td>Cyclist</td></tr><tr><td>Baseline</td><td>66.4/66.0</td><td>73.5/67.4</td><td>70.4/69.3</td><td>70.1/67.6</td></tr><tr><td>Placement 1</td><td>66.5/66.1</td><td>74.8/69.1</td><td>71.1/70.2</td><td>70.1/68.6</td></tr><tr><td>Placement 2 (Ours)</td><td>67.0/66.6</td><td>75.4/70.2</td><td>71.9/71.0</td><td>71.4/69.3</td></tr></table>

Table 10: The ratio $r$ in voxel genenration.

<table><tr><td rowspan="2">Ratio</td><td colspan="3">3D AP/APH (L2)</td><td rowspan="2">mAP/mAPH (L2)</td></tr><tr><td>Vehicle</td><td>Pedestrian</td><td>Cyclist</td></tr><tr><td>0</td><td>66.5/66.1</td><td>74.8/69.6</td><td>70.9/70.0</td><td>70.8/68.6</td></tr><tr><td>0.2 (Ours)</td><td>67.0/66.6</td><td>75.4/70.2</td><td>71.9/71.0</td><td>71.4/69.3</td></tr><tr><td>0.5</td><td>67.2/66.8</td><td>75.3/70.0</td><td>72.1/71.1</td><td>71.5/69.3</td></tr></table>

<!-- Media -->

The Ratio in Voxel Generation. We conduct the ablation study for foreground selection ratio $r$ in voxel generation. As shown in Table 10,compared with baseline $\left( {r = 0}\right)$ ,the manner of setting $r = {0.2}$ brings ${0.7}\mathrm{{mAPH}}/\mathrm{L}2$ performance improvement. When we set a larger ratio $r = {0.5}$ , the performance is improved slightly. Therefore,we set $r = {0.2}$ to balance the performance and computation cost.

Different Window Sizes and Group Sizes in Inference. To analyze the impact of window size and group size in the inference process, we evaluate the results under the cases of different window sizes and group sizes with the same trained model of LION-Mamba (i.e.,window size=\{(13,13,32), $\left( {{13},{13},{16}}\right) ,\left( {{13},{13},8}\right)$ ,and $\left( {{13},{13},4}\right) \}$ and group size=\{4096,2048,1024,512\}) on WOD 100% training data. As shown in Table 11, surprisingly, we find that using different window sizes or group sizes during inference still does not significantly affect performance. This indicates that LION might decrease the strong dependence on hand-crafted priors and have good extrapolation ability.

### A.3 Comparison of Computation Cost and Parameter Size

We compare different 3D backbones on the WOD validation set. As shown in Table 12, LION with RWKV [38], RetNet [52], and Mamba [22], achieve 72.8, 72.7, and 73.2 mAPH/L2, respectively. Compared with other sparse convolution and transformer-based backbones, LION achieves superior accuracy while maintaining a satisfactory computation cost and parameter size.

<!-- Media -->

Table 11: Comparison of different window and group sizes in inference on WOD validation set (train with ${100}\%$ training data). Bold denotes the result of LION with the default settings in the main paper.

<table><tr><td>Window Size</td><td>Group Size</td><td>mAP/mAPH (L2)</td></tr><tr><td>$\left\lbrack  {7,7,{32}}\right\rbrack  ,\left\lbrack  {7,7,{16}}\right\rbrack  ,\left\lbrack  {7,7,8}\right\rbrack  ,\left\lbrack  {7,7,4}\right\rbrack$</td><td>$\left\lbrack  {{4096},{2048},{1024},{512}}\right\rbrack$</td><td>73.24</td></tr><tr><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack  ,\left\lbrack  {{13},{13},{16}}\right\rbrack  ,\left\lbrack  {{13},{13},8}\right\rbrack  ,\left\lbrack  {{13},{13},4}\right\rbrack$</td><td>[4096, 2048, 1024, 512]</td><td>73.24</td></tr><tr><td>$\left\lbrack  {{25},{25},{32}}\right\rbrack  ,\left\lbrack  {{25},{25},{16}}\right\rbrack  ,\left\lbrack  {{25},{25},8}\right\rbrack  ,\left\lbrack  {{25},{25},4}\right\rbrack$</td><td>[4096, 2048, 1024, 512]</td><td>73.25</td></tr><tr><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack  ,\left\lbrack  {{13},{13},{16}}\right\rbrack  ,\left\lbrack  {{13},{13},8}\right\rbrack  ,\left\lbrack  {{13},{13},4}\right\rbrack$</td><td>$\left\lbrack  {{2048},{1024},{512},{256}}\right\rbrack$</td><td>73.18</td></tr><tr><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack  ,\left\lbrack  {{13},{13},{16}}\right\rbrack  ,\left\lbrack  {{13},{13},8}\right\rbrack  ,\left\lbrack  {{13},{13},4}\right\rbrack$</td><td>[4096, 2048, 1024, 512]</td><td>73.24</td></tr><tr><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack  ,\left\lbrack  {{13},{13},{16}}\right\rbrack  ,\left\lbrack  {{13},{13},8}\right\rbrack  ,\left\lbrack  {{13},{13},4}\right\rbrack$</td><td>$\left\lbrack  {{8192},{4096},{2048},{1024}}\right\rbrack$</td><td>73.02</td></tr></table>

<!-- Meanless: 16 -->


Table 12: Comparison of computation cost and parameter size of different 3D backbones on the WOD validation set.

<table><tr><td>Backbone</td><td>Operator</td><td>mAP/mAPH (L2)</td><td>FLOPs (G)</td><td>Params (M)</td></tr><tr><td>ResBackbone8× [66]</td><td>Sparse Conv</td><td>68.2/65.8</td><td>48.2</td><td>2.7</td></tr><tr><td>SST [15]</td><td>Transformer</td><td>67.8/64.6</td><td>86.2</td><td>1.6</td></tr><tr><td>FlatFormer [32]</td><td>Transformer</td><td>69.7/67.2</td><td>48.3</td><td>1.1</td></tr><tr><td>DSVT-Pillar [57]</td><td>Transformer</td><td>73.2/71.0</td><td>88.2</td><td>2.7</td></tr><tr><td>DSVT-Voxel [57]</td><td>Transformer</td><td>74.0/72.1</td><td>100.8</td><td>2.7</td></tr><tr><td>LION (Ours)</td><td>RWKV</td><td>74.7/72.8</td><td>79.4</td><td>3.0</td></tr><tr><td>LION (Ours)</td><td>RetNet</td><td>74.6/72.7</td><td>51.3</td><td>2.0</td></tr><tr><td>LION (Ours)</td><td>Mamba</td><td>75.1/73.2</td><td>58.5</td><td>1.4</td></tr></table>

Table 13: Detailed architecture specifications on Waymo Open dataset.

<table><tr><td></td><td>LION-RWKV</td><td>LION-RetNet</td><td>LION-Mamba</td><td>LION-Mamba-L</td></tr><tr><td>Block</td><td>Window Shape <br> Dim, Group Size</td><td>Window Shape <br> Dim, Group Size</td><td>Window Shape <br> Dim, Group Size</td><td>Window Shape <br> Dim, Group Size</td></tr><tr><td>Block 1</td><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack$ <br> 64,4096</td><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack$ <br> 64,4096</td><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack$ <br> 64,4096</td><td>$\left\lbrack  {{13},{13},{32}}\right\rbrack$ <br> 128,4096</td></tr><tr><td>Block 2</td><td>$\left\lbrack  {{13},{13},{16}}\right\rbrack$ <br> 64, 2048</td><td>$\left\lbrack  {{13},{13},{16}}\right\rbrack$ <br> 64, 2048</td><td>$\left\lbrack  {{13},{13},{16}}\right\rbrack$ <br> 64, 2048</td><td>$\left\lbrack  {{13},{13},{16}}\right\rbrack$ <br> 128, 2048</td></tr><tr><td>Block 3</td><td>$\left\lbrack  {{13},{13},8}\right\rbrack$ <br> 64, 1024</td><td>$\left\lbrack  {{13},{13},8}\right\rbrack$ <br> 64, 1024</td><td>$\left\lbrack  {{13},{13},8}\right\rbrack$ <br> 64, 1024</td><td>$\left\lbrack  {{13},{13},8}\right\rbrack$ <br> 128, 1024</td></tr><tr><td>Block 4</td><td>$\left\lbrack  {{13},{13},4}\right\rbrack$ <br> 64,512</td><td>$\left\lbrack  {{13},{13},4}\right\rbrack$ <br> 64,512</td><td>$\left\lbrack  {{13},{13},4}\right\rbrack$ <br> 64,512</td><td>$\left\lbrack  {{13},{13},4}\right\rbrack$ <br> 128, 512</td></tr></table>

<!-- Media -->

### A.4 Architecture Specifications

As shown in Table 13, the architecture specifications of the LION models (LION-RWKV, LION-RetNet, LION-Mamba, and LION-Mamba-L) on Waymo Open dataset are detailed in terms of window shape, dimension, and group size. For LION-Mamba-L, we set the dimension to 128 to double the channel of LION.

### A.5 Visualization for Feature Map

As shown in Figure 6, we visualize feature maps of different LION blocks. We can observe that as the features pass through more blocks, the magnitude of the foreground's feature response becomes larger, demonstrating the rationality of distinguishing foreground voxels by feature response. Besides, we find that the foreground features become more dense and more distinguished, which also demonstrates the effectiveness of the voxel generation operation.

### A.6 Comparison of Qualitative Results with DSVT

To illustrate the superiority of LION, we present the visualization of the qualitative results of DSVT [57] (a) and LION (b) on the WOD [49] validation set, as shown in Figure 7. Specifically, in the first and third columns, our LION can reduce more false positives compared with DSVT. In the second column, our LION even detects some hard objects at a distance. In the last column, our LION can achieve more accurate localization. These qualitative results demonstrate the superior performance of our LION.

<!-- Meanless: 17 -->


<!-- Media -->

<!-- figureText: High<br>...<br>...<br>、工<br>：___<br>...<br>Research<br>-<br>8<br>净含<br>Low<br>Block 1<br>Block 2<br>Block 3<br>Block 4 -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_17.jpg?x=307&y=198&w=1182&h=717&r=0"/>

Figure 6: Visualization of feature map of different blocks. We highlight the foreground annotated by red GT boxes. The color map represents the magnitude of the feature response.

<!-- figureText: P<br>(a)<br>(b) -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_17.jpg?x=312&y=1013&w=1179&h=517&r=0"/>

Figure 7: Comparison of DSVT and LION on the WOD validation set from the BEV perspective. Blue and green boxes are the prediction and ground truth boxes. It can be seen that LION can achieve better results compared to DSVT, demonstrating the superiority of LION.

<!-- Media -->

### A.7 Qualitative Results

As shown in Figure 8, we visualize the qualitative results of LION on the WOD validation set. As shown in the first column, LION can still achieve satisfactory results even in crowded 3D scenes. However, as shown in the second and third columns, LION misses some objects at a distance with sparse point clouds. Therefore, we will further improve the performance of distant objects by fusing the image features in the future.

### A.8 Broader Impacts

LION achieves promising performance for 3D object detection, enhancing the safety of autonomous driving. However, LION has relatively high requirements on computing resources to achieve faster running speed, which puts forward higher requirements for the hardware of autonomous driving.

<!-- Meanless: 18 -->


<!-- Media -->

<!-- figureText: (a)<br>(b)<br>(c) -->

<img src="https://cdn.noedgeai.com/bo_d7tder2lb0pc73f30rg0_18.jpg?x=311&y=197&w=1180&h=354&r=0"/>

Figure 8: Qualitative results of LION on the WOD validation set. Green and blue boxes denote ground truth and predicted bounding boxes, respectively.

<!-- Media -->

Future research could focus on optimizing LION to improve bottlenecks in running speed while maintaining high detection accuracy, making it more accessible and practical for autonomous driving.

<!-- Meanless: 19 -->