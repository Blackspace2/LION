**把高维空间中的点映射到一维序列，同时尽量保持局部邻近关系。**

# 谱图带宽感知空间描述符：Spectral Bandwidth-Aware Spatial Descriptor

简称可以叫：

**SBSD：Spectral Bandwidth-aware Spatial Descriptor**

它比我上一个结构张量方案更数学化。核心思想是：

**不要只问 voxel 的 3D 坐标是什么，而是问当前 1D 序列排列对 3D 邻接图造成了多大带宽损失，并用谱图信号补偿这种损失。**

这直接对应 LION 说的问题：3D 空间近邻在 1D 序列中可能距离很远。Fore-Mamba3D 也明确指出 3D→1D sequence 会造成 geometric distortion，因此需要额外的 state spatial fusion 来补偿。

---

## 1. 把 voxel 建成局部图

对一个 LION group 内的 voxel：

[
\mathcal{V}*g={(p_i,x_i)}*{i=1}^{N_g}
]

建立邻接图：

[
G_g=(V_g,E_g)
]

如果两个 voxel 在 3D 空间中足够近，就连边：

[
(i,j)\in E_g \quad \Longleftrightarrow \quad |p_i-p_j|_2 \le r
]

或者用 3×3×3 邻域 / kNN。边权为：

[
A_{ij}=\exp\left(-\frac{|p_i-p_j|_2^2}{2\sigma^2}\right)
]

得到邻接矩阵 (A)，度矩阵：

[
D_{ii}=\sum_j A_{ij}
]

图拉普拉斯：

[
L=D-A
]

归一化拉普拉斯也可以：

[
L_{norm}=I-D^{-\frac12}AD^{-\frac12}
]

这个图就是“真实 3D 空间邻接关系”。

---

## 2. 序列排列造成的带宽损失

LION / Mamba 会把 group 内 voxel 排成一个序列。设排序函数为：

[
r(i)\in {1,\dots,N_g}
]

表示 voxel (i) 在 1D 序列中的位置。

对每个节点定义局部带宽畸变：

[
b_i=\max_{j\in \mathcal{N}(i)} |r(i)-r(j)|
]

或者用更平滑的加权平均：

[
\bar{b}*i=
\frac{
\sum*{j\in \mathcal{N}(i)} A_{ij}|r(i)-r(j)|
}{
\sum_{j\in \mathcal{N}(i)} A_{ij}+\epsilon
}
]

这个量非常关键。

它直接衡量：

**3D 中和我相邻的 voxel，在 1D 序列中被拉开了多远。**

如果 (\bar{b}_i) 大，说明当前 scanning order 对这个 voxel 很不友好，Linear RNN 很可能难以局部建模。

---

## 3. 谱图位置坐标

再做一个谱图描述。取图拉普拉斯最小的几个非平凡特征向量：

[
L u_k=\lambda_k u_k
]

取：

[
s_i=[u_1(i),u_2(i),...,u_K(i)]
]

这个 (s_i) 是 voxel 在图谱空间里的坐标。

直观理解：

普通坐标 (p_i=(x,y,z)) 表示欧氏空间位置。

谱图坐标 (s_i) 表示它在局部邻接图结构中的位置。

如果两个 voxel 在图上强连接，它们的谱坐标会接近。Laplacian Eigenmaps 的基本目标正是让图上邻近点在低维表示里保持接近。([神经信息处理系统会议论文][3])

但是注意：每个 group 都做 eigen decomposition 可能慢。工程上不建议第一版真做特征分解。可以用 **Chebyshev / Laplacian polynomial filter** 近似谱特征。

---

## 4. 不做特征分解的工程版本

为了实用，可以不显式求 eigenvectors，而是用图拉普拉斯多项式构造描述符。

定义原始几何信号：

[
P=[p_1,p_2,\dots,p_{N_g}]^T\in \mathbb{R}^{N_g\times 3}
]

做一阶图平滑残差：

[
R^{(1)}=LP
]

二阶图平滑残差：

[
R^{(2)}=L^2P
]

其中：

[
R_i^{(1)} = \sum_j A_{ij}(p_i-p_j)
]

它表示当前 voxel 相对于邻域的局部几何不平衡程度。

再定义谱能量：

[
e_i^{(1)}=|R_i^{(1)}|_2
]

[
e_i^{(2)}=|R_i^{(2)}|_2
]

这两个量可以理解为局部图信号的高频强度。目标边缘、稀疏断裂、几何突变处通常更高；平滑表面内部更低。

最后 descriptor：

[
d_i=
[
\bar{b}_i,\
e_i^{(1)},\
e_i^{(2)},\
\rho_i,\
\Delta p_i
]
]

其中 (\rho_i) 是邻域密度，(\Delta p_i) 是局部质心偏移。

---


# 这个方法为什么更“数学”

它背后不是经验性加坐标，而是三个明确数学问题：

第一，**图带宽最小化**：

[
\min_{\pi}\max_{(i,j)\in E}|r_\pi(i)-r_\pi(j)|
]

这个目标正是“让 3D 邻居在 1D 序列中尽量靠近”。它和 sparse matrix reordering 是同一个数学问题。RCM 等算法就是为了让图/矩阵重排序后带宽更小。([EECS Berkeley][2])

第二，**拉普拉斯局部保持**：

[
\min_Y \sum_{i,j} A_{ij}|Y_i-Y_j|^2
]

这个目标是 Laplacian Eigenmaps 的核心思想：邻接图中相似的点，在嵌入空间中也应该相近。([神经信息处理系统会议论文][3])

第三，**图信号高频能量**：

[
E(X)=\text{Tr}(X^T L X)
=======================

\frac{1}{2}\sum_{i,j}A_{ij}|x_i-x_j|^2
]

这个能量衡量图上信号是否平滑。你可以把 voxel 坐标、特征、ground residual 都看作图信号。高频大，说明局部结构变化剧烈，Mamba 不能简单依赖顺序递推。

这比普通 PE 更容易写出严谨公式，也更容易在论文里讲清楚“为什么它针对 3D→1D 序列化问题”。

---

# 建议消融实验

| 实验                    | 改动                                  |
| --------------------- | ----------------------------------- |
| Baseline LION         | 原始 descriptor                       |
| LION + SBSD           | 仅替换 descriptor                      |
| LION + Bandwidth-only | 只保留 (\bar{b}_i)                     |
| LION + Spectral-only  | 只保留 (e_i^{(1)},e_i^{(2)})           |
| LION + SBSD full      | (\bar{b}_i + e_i^{(1)} + e_i^{(2)}) |

