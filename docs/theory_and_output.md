# 震源机制反演原理与输出格式详解

## 目录

- [1. 贝叶斯反演理论基础](#1-贝叶斯反演理论基础)
- [2. 矩张量与参数化表示](#2-矩张量与参数化表示)
- [3. 似然函数](#3-似然函数)
- [4. 不确定性处理](#4-不确定性处理)
- [5. 搜索算法](#5-搜索算法)
- [6. 输出内容与文件格式](#6-输出内容与文件格式)
- [7. 参考文献](#7-参考文献)

---

## 1. 贝叶斯反演理论基础

### 1.1 核心思想

MTfit 使用**正演模型贝叶斯反演**方法来求解地震震源机制。其核心思想是：在矩张量参数空间中大量随机采样，对每个采样计算其产生观测数据的概率（似然），从而构建矩张量的后验概率密度函数（PDF）。

### 1.2 贝叶斯定理

$$
P(\mathbf{M} | \mathbf{d}) = \frac{P(\mathbf{d} | \mathbf{M}) \cdot P(\mathbf{M})}{P(\mathbf{d})}
$$

其中：
- $P(\mathbf{M} | \mathbf{d})$ — **后验概率**：给定观测数据 $\mathbf{d}$ 时矩张量 $\mathbf{M}$ 的概率
- $P(\mathbf{d} | \mathbf{M})$ — **似然函数**：给定矩张量时产生观测数据的概率
- $P(\mathbf{M})$ — **先验概率**：矩张量的先验分布（默认在 6 维球面上均匀采样）
- $P(\mathbf{d})$ — **贝叶斯证据**（归一化常数）

### 1.3 正演模型

给定矩张量 $\mathbf{M}$ 和台站 $i$ 的辐射系数矩阵 $\mathbf{a}_i$（由方位角和离源角决定），理论振幅为：

$$
X_i = \mathbf{a}_i \cdot \mathbf{M}
$$

其中 $\mathbf{a}_i$ 是一个 $1 \times 6$ 的系数向量，包含了台站位置和波类型（P、SH、SV）的信息：

$$
\mathbf{a}_i = [a_{xx}, a_{yy}, a_{zz}, \sqrt{2} a_{xy}, \sqrt{2} a_{xz}, \sqrt{2} a_{yz}]_i
$$

系数根据不同波型计算：
- **P 波**：由径向辐射花样决定
- **SH 波**：由横向剪切辐射花样决定
- **SV 波**：由纵向剪切辐射花样决定

### 1.4 贝叶斯证据

贝叶斯证据（边际似然）通过蒙特卡洛积分估计：

$$
Z = P(\mathbf{d}) = \int P(\mathbf{d} | \mathbf{M}) P(\mathbf{M}) \, d\mathbf{M} \approx \frac{1}{N} \sum_{i=1}^{N} P(\mathbf{d} | \mathbf{M}_i)
$$

其中 $\mathbf{M}_i$ 从先验分布 $P(\mathbf{M})$ 中采样。贝叶斯证据可用于模型比较，例如比较双力偶模型和全矩张量模型哪个更好地解释数据。

### 1.5 Kullback-Leibler 散度

KL 散度用于衡量后验分布相对于先验分布所包含的信息量：

$$
D_{KL} = \int P(\mathbf{M} | \mathbf{d}) \ln \frac{P(\mathbf{M} | \mathbf{d})}{P(\mathbf{M})} \, d\mathbf{M}
$$

$D_{KL}$ 值越大，表示数据对矩张量的约束越强。

---

## 2. 矩张量与参数化表示

### 2.1 矩张量 3×3 矩阵

地震矩张量是一个 $3 \times 3$ 的对称矩阵，在北(X)—东(Y)—下(Z) 坐标系中表示为：

$$
\mathbf{M}_{33} = \begin{pmatrix} M_{xx} & M_{xy} & M_{xz} \\ M_{xy} & M_{yy} & M_{yz} \\ M_{xz} & M_{yz} & M_{zz} \end{pmatrix}
$$

### 2.2 矩张量 6 分量向量

为保持归一化，MTfit 使用修正的 6 分量向量表示：

$$
\mathbf{M}_6 = \begin{pmatrix} M_{xx} \\ M_{yy} \\ M_{zz} \\ \sqrt{2} M_{xy} \\ \sqrt{2} M_{xz} \\ \sqrt{2} M_{yz} \end{pmatrix}
$$

$\sqrt{2}$ 因子确保 6 向量的模等于 3×3 矩阵的 Frobenius 范数。

MTfit 中所有矩张量均归一化为单位向量（$|\mathbf{M}_6| = 1$），因此反演结果只包含震源机制信息，不包含绝对标量地震矩。

### 2.3 T-N-P 轴与特征值

对 $\mathbf{M}_{33}$ 进行特征分解：

$$
\mathbf{M}_{33} = \lambda_1 \mathbf{t}\mathbf{t}^T + \lambda_2 \mathbf{n}\mathbf{n}^T + \lambda_3 \mathbf{p}\mathbf{p}^T
$$

其中：
- $\lambda_1 \geq \lambda_2 \geq \lambda_3$ — 特征值（按降序排列）
- $\mathbf{t}$ — **T 轴**（拉张轴，最大特征值方向）
- $\mathbf{n}$ — **N 轴**（中间轴）
- $\mathbf{p}$ — **P 轴**（压缩轴，最小特征值方向）

### 2.4 走向、倾角、滑动角 (Strike, Dip, Rake)

由 T-N-P 轴可以计算两个共轭断层面的参数：

| 参数 | 符号 | 范围 | 含义 |
|------|------|------|------|
| 走向 (Strike) | $\phi_s$ | $[0°, 360°)$ | 断层面与正北方向的夹角 |
| 倾角 (Dip) | $\delta$ | $[0°, 90°]$ | 断层面与水平面的夹角 |
| 滑动角 (Rake) | $\lambda$ | $[-180°, 180°]$ | 滑动方向在断层面上的角度 |

输出中 `S1, D1, R1` 为断层面 1 的参数，`S2, D2, R2` 为辅助面（断层面 2）的参数。

### 2.5 Tape 参数 (γ, δ, κ, h, σ)

基于 Tape & Tape (2012, 2013) 的参数化，将矩张量分解为震源类型和断层方位两组参数：

| 参数 | 符号 | 范围 | 物理含义 |
|------|------|------|---------|
| gamma | $\gamma$ | $[-\pi/6, \pi/6]$ | **震源类型角**（位于基本矩张量 lune 上的经度），$\gamma=0$ 为纯双力偶 |
| delta | $\delta$ | $[-\pi/2, \pi/2]$ | **震源类型角**（lune 上的纬度），$\delta=0$ 为纯偏差张量，$\delta=\pm\pi/2$ 为纯各向同性 |
| kappa | $\kappa$ | $[0, 2\pi)$ | **走向角**（弧度） |
| h | $h$ | $[0, 1]$ | $\cos(\text{dip})$，倾角的余弦值 |
| sigma | $\sigma$ | $[-\pi/2, \pi/2]$ | **滑动角**（弧度） |

**特殊震源类型位置：**
- $(\gamma, \delta) = (0, 0)$ → 纯双力偶 (DC)
- $(\gamma, \delta) = (\pm\pi/6, 0)$ → 纯 CLVD
- $(\gamma, \delta) = (0, \pm\pi/2)$ → 纯各向同性（爆炸/内爆）

### 2.6 Hudson 参数 (u, v)

基于 Hudson et al. (1989) 的参数化，将矩张量源类型映射到二维图（Hudson 图）上：

| 参数 | 范围 | 物理含义 |
|------|------|---------|
| u | $[-1, 1]$ | 与各向同性分量相关，由特征值 $k = (\lambda_1+\lambda_2+\lambda_3)/3$ 导出 |
| v | $[-1, 1]$ | 与 CLVD 分量相关，由特征值偏差部分导出 |

**Hudson 图中的关键位置：**
- $(u, v) = (0, 0)$ → 纯双力偶
- $(u, v) = (0, \pm1)$ → 纯 CLVD
- $(u, v) = (\pm1, 0)$ → 纯各向同性

---

## 3. 似然函数

### 3.1 P 波极性似然函数

对于 P 波极性观测（+1 表示向上初动，-1 表示向下初动），似然函数定义为：

$$
P(p_i^{obs} | \mathbf{M}, \sigma_i) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{p_i^{obs} \cdot X_i}{\sqrt{2}\,\sigma_i}\right)\right] (1-\epsilon_i) + \frac{1}{2}\left[1 + \text{erf}\left(\frac{-p_i^{obs} \cdot X_i}{\sqrt{2}\,\sigma_i}\right)\right] \epsilon_i
$$

其中：
- $p_i^{obs}$ — 观测极性（+1 或 -1）
- $X_i = \mathbf{a}_i \cdot \mathbf{M}$ — 理论振幅
- $\sigma_i$ — **分数不确定性**（fractional uncertainty），描述振幅测量的相对误差
- $\epsilon_i$ — **极性误判概率**（incorrect polarity probability），考虑台站方位误差导致的极性反转
- $\text{erf}$ — 误差函数

**物理解释**：当理论振幅 $X_i$ 与观测极性 $p_i^{obs}$ 符号一致时（即模型预测正确），误差函数值趋向 1，概率接近 $(1-\epsilon_i)$。当不一致时，概率接近 $\epsilon_i$（误判概率）。

### 3.2 多台站联合概率

假设各台站观测独立，所有台站的联合对数概率为各台站对数概率之和：

$$
\ln P(\mathbf{d} | \mathbf{M}) = \sum_{i=1}^{N_s} \ln P(p_i^{obs} | \mathbf{M}, \sigma_i)
$$

### 3.3 振幅比似然函数

对于振幅比观测 $r = A_x / A_y$，似然函数基于双变量对数正态分布：

$$
P(r | \mathbf{M}, \sigma_x, \sigma_y) = P_{\text{ratio}}(r, \mu_x, \mu_y, s_x, s_y) + P_{\text{ratio}}(-r, \mu_x, \mu_y, s_x, s_y)
$$

其中：
- $\mu_x = \mathbf{a}_x \cdot \mathbf{M}$, $\mu_y = \mathbf{a}_y \cdot \mathbf{M}$ — 分子和分母的理论振幅
- $s_x = |\mu_x| \cdot \sigma_{x\%}$, $s_y = |\mu_y| \cdot \sigma_{y\%}$ — 标准差（由百分比误差乘以理论振幅得到）
- 第二项 $P_{\text{ratio}}(-r, ...)$ 处理负振幅比的情况

**支持的振幅比类型：**
- P/SH 均方根振幅比
- P/SV 均方根振幅比
- SH/SV 均方根振幅比

---

## 4. 不确定性处理

### 4.1 震源位置不确定性

MTfit 通过蒙特卡洛积分边缘化震源位置不确定性：

$$
P(\mathbf{M} | \mathbf{d}) = \int P(\mathbf{M} | \mathbf{d}, \mathbf{x}) \cdot P(\mathbf{x} | \mathbf{d}) \, d\mathbf{x}
$$

其中 $P(\mathbf{x} | \mathbf{d})$ 是位置后验概率（来自定位程序如 NonLinLoc 的 scatter 文件），积分通过对位置样本求和实现：

$$
P(\mathbf{M} | \mathbf{d}) \approx \sum_{j=1}^{N_{loc}} P(\mathbf{M} | \mathbf{d}, \mathbf{x}_j) \cdot w_j \cdot dV_j
$$

其中 $w_j$ 是位置样本 $j$ 的权重，$dV_j$ 是对应的体积元。

**实现方式**：位置样本通过 `.scatangle` 文件提供，每个位置样本包含所有台站的方位角和离源角。程序对每个位置样本分别计算辐射花样系数，然后对位置进行加权求和。

### 4.2 速度模型不确定性

速度模型的不确定性同样通过位置 PDF 间接处理，因为不同速度模型会导致不同的射线路径和离源角。

### 4.3 多事件联合反演

对于 $K$ 个事件的联合反演，后验概率为各事件后验的乘积：

$$
P(\mathbf{M}_1, ..., \mathbf{M}_K | \mathbf{d}_1, ..., \mathbf{d}_K) = \prod_{k=1}^{K} P(\mathbf{M}_k | \mathbf{d}_k)
$$

联合反演可以利用事件间的相对振幅信息，对相对尺度因子进行估计。

---

## 5. 搜索算法

### 5.1 迭代随机采样 (`iterate`)

在矩张量空间中均匀随机采样，计算每个样本的似然值。适用于快速获得概率密度函数的初步估计。

**参数**：
- `max_samples`：最大采样数（例如 1,000,000）
- `number_samples`：每次迭代的采样数（由内存限制决定）

**采样方式**：
- **全矩张量空间**：在 6 维超球面上均匀采样（6-sphere prior）
- **双力偶空间**：在双力偶子空间上均匀采样

### 5.2 时间限制采样 (`time`)

与迭代采样相同的算法，但以时间作为终止条件。

**参数**：
- `max_time`：最大运行时间（秒）

### 5.3 马尔可夫链蒙特卡洛 (`mcmc`)

使用 Metropolis-Hastings 算法在矩张量空间中进行采样，适用于需要更高效探索高概率区域的情况。

**算法流程**：
1. 从初始点开始
2. 按提议分布生成候选样本
3. 计算接受概率：$\alpha = \min(1, P(\mathbf{d}|\mathbf{M}_{new}) / P(\mathbf{d}|\mathbf{M}_{old}))$
4. 以概率 $\alpha$ 接受新样本
5. 重复直到链收敛

### 5.4 跨维 MCMC (`transdmcmc`)

允许在双力偶空间和全矩张量空间之间跳转的 MCMC 算法，通过可逆跳转实现模型选择。

---

## 6. 输出内容与文件格式

### 6.1 输出数据总览

反演完成后，输出数据包含以下核心内容：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `MTSpace` | float64 | (6, N) | 非零概率的矩张量 6 向量样本 |
| `Probability` | float64 | (N,) | 归一化后的概率值 |
| `ln_pdf` | float64 | (N,) | 未归一化的对数概率密度 |
| `NSamples` | int | 标量 | 总采样数（包括零概率样本） |
| `dV` | float64 | 标量 | 采样的体积元 |
| `ln_bayesian_evidence` | float64 | 标量 | 对数贝叶斯证据 |

**当设置 `convert=True` 时，额外输出参数化结果：**

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `g` | float64 | (N,) | Tape gamma 参数（弧度） |
| `d` | float64 | (N,) | Tape delta 参数（弧度） |
| `k` | float64 | (N,) | Tape kappa / 走向角（弧度） |
| `h` | float64 | (N,) | cos(dip)，倾角余弦 |
| `s` | float64 | (N,) | Tape sigma / 滑动角（弧度） |
| `u` | float64 | (N,) | Hudson u 参数 |
| `v` | float64 | (N,) | Hudson v 参数 |
| `S1` | float64 | (N,) | 断层面 1 走向（度） |
| `D1` | float64 | (N,) | 断层面 1 倾角（度） |
| `R1` | float64 | (N,) | 断层面 1 滑动角（度） |
| `S2` | float64 | (N,) | 断层面 2 走向（度） |
| `D2` | float64 | (N,) | 断层面 2 倾角（度） |
| `R2` | float64 | (N,) | 断层面 2 滑动角（度） |

### 6.2 MATLAB 格式 (.mat)

MATLAB 格式使用 `scipy.io.savemat`（v7）或 `hdf5storage.savemat`（v7.3）保存。

**文件结构：**

```
主文件：{basename}MT.mat 或 {basename}DC.mat
├── Events                      # 主要反演结果
│   ├── NSamples               # int: 总采样数
│   ├── dV                     # float: 体积元
│   ├── Probability            # array (1, N): 归一化概率
│   ├── MTSpace                # array (6, N): 矩张量样本
│   ├── ln_pdf                 # array (1, N): 对数概率密度
│   ├── ln_bayesian_evidence   # float: 对数贝叶斯证据
│   ├── g                      # array (N,): Tape gamma
│   ├── d                      # array (N,): Tape delta
│   ├── k                      # array (N,): Tape kappa
│   ├── h                      # array (N,): cos(dip)
│   ├── s                      # array (N,): Tape sigma
│   ├── u                      # array (N,): Hudson u
│   ├── v                      # array (N,): Hudson v
│   ├── S1, D1, R1             # array (N,): 断层面 1 参数（度）
│   ├── S2, D2, R2             # array (N,): 断层面 2 参数（度）
│   └── UID                    # string: 事件标识
├── Stations                    # 台站信息
│   └── array (N_sta, 4)       # [名称, 方位角, 离源角, 观测值]
└── Other                       # 辅助信息
    ├── Inversions             # list: 使用的数据类型
    ├── a_polarity             # array: 极性辐射系数
    ├── error_polarity         # array: 极性误差
    ├── a1_amplitude_ratio     # array: 振幅比分子系数
    ├── a2_amplitude_ratio     # array: 振幅比分母系数
    ├── percentage_error1_...  # array: 分子百分比误差
    └── percentage_error2_...  # array: 分母百分比误差

台站分布文件：{basename}StationDistribution.mat（如果有位置 PDF）
├── StationDistribution
│   ├── Distribution           # array: 位置样本
│   └── Probability            # array: 位置样本权重
```

### 6.3 Pickle 格式 (.out)

Python pickle 格式，直接保存 Python 字典。数据结构与 MATLAB 格式完全相同。

**文件命名：**
- 主文件：`{basename}MT.out` 或 `{basename}DC.out`
- 台站分布：`{basename}StationDistribution.out`

**读取方式：**
```python
import pickle
with open('MTfitOutputMT.out', 'rb') as f:
    result = pickle.load(f)

# 访问矩张量样本
mt_samples = result['Events']['MTSpace']  # shape: (6, N)
probability = result['Events']['Probability']  # shape: (1, N)

# 获取最大概率解
best_idx = probability.argmax()
best_mt = mt_samples[:, best_idx]
```

### 6.4 HYP 格式 (.hyp + .mt + .sf)

NonLinLoc 兼容的文本格式，包含三个文件：

**`.hyp` 文件**（文本格式）：
```
NLLOC ... (定位信息头)
GEOGRAPHIC ... (震源位置)
STATISTICS ... (位置不确定性统计)
PHASE ... (台站数据: 名称, 方位角, 离源角, 极性, 走时等)
FOCALMECH ... (最佳拟合走向/倾角/滑动角)
END_NLLOC
```

**`.mt` 文件**（二进制格式）：
- 包含所有非零概率矩张量样本的二进制编码
- 每个样本 6 个 float64 值 + 1 个概率值

**`.sf` 文件**（二进制格式）：
- 包含振幅反演的尺度因子
- 仅在进行振幅比反演时生成

### 6.5 输出文件命名规则

默认输出基名为 `MTfitOutput`，可通过 `-o` 参数自定义。

| 反演类型 | MATLAB 文件 | Pickle 文件 |
|---------|------------|------------|
| 全矩张量 | `{basename}MT.mat` | `{basename}MT.out` |
| 双力偶 | `{basename}DC.mat` | `{basename}DC.out` |

### 6.6 如何读取和使用输出

#### Python 读取 MATLAB 输出

```python
import scipy.io as sio

# 读取 v7 格式
data = sio.loadmat('MTfitOutputMT.mat')

# 读取 v7.3 格式（需要 h5py）
import h5py
with h5py.File('MTfitOutputMT.mat', 'r') as f:
    mt_space = f['Events']['MTSpace'][:]
    prob = f['Events']['Probability'][:]
```

#### 提取最佳解

```python
import numpy as np

# 最大概率解
best_idx = np.argmax(probability)
best_mt6 = mt_space[:, best_idx]

# 最大概率解的断层面参数（如果已转换）
best_strike = S1[best_idx]  # 度
best_dip = D1[best_idx]     # 度
best_rake = R1[best_idx]    # 度
```

#### 计算均值矩张量

```python
# 概率加权均值
mean_mt = np.sum(mt_space * probability, axis=1)
mean_mt = mean_mt / np.linalg.norm(mean_mt)  # 归一化
```

---

## 7. 参考文献

1. **Pugh, D. J., White, R. S., & Christie, P. A. F. (2016).** A Bayesian method for microseismic source inversion. *Geophysical Journal International*, 206(2), 1009-1038.

2. **Pugh, D. J. (2015).** *Bayesian Source Inversion of Microseismic Events*. PhD Thesis, Department of Earth Sciences, University of Cambridge.

3. **Tape, W. & Tape, C. (2012).** A geometric setting for moment tensors. *Geophysical Journal International*, 190(1), 499-514.

4. **Tape, W. & Tape, C. (2013).** The classical model for moment tensors. *Geophysical Journal International*, 195(3), 1701-1720.

5. **Hudson, J. A., Pearce, R. G., & Rogers, R. M. (1989).** Source type plot for inversion of the moment tensor. *Journal of Geophysical Research*, 94(B1), 765-774.
