# MTfit - 贝叶斯矩张量反演程序

基于正演模型的贝叶斯矩张量反演代码，用于地震学中的震源机制解反演。

本项目是对 [djpugh/MTfit](https://github.com/djpugh/MTfit) 的 Python 3.13 现代化重构版本。

## 项目简介

MTfit 使用贝叶斯方法对地震矩张量进行反演，基于 Pugh et al. (2016) 提出的方法。程序通过蒙特卡洛采样在矩张量空间中搜索最优解，并利用贝叶斯定理边缘化各种不确定性（测量误差、位置不确定性、模型不确定性等）。

### 核心功能

- **P 波极性反演** - 利用初至 P 波极性数据约束震源机制
- **振幅比反演** - 利用 P/SH 振幅比数据进行更精确的反演
- **双力偶约束反演** - 将搜索空间限制在双力偶解空间
- **全矩张量反演** - 在完整的矩张量空间中搜索
- **多事件联合反演** - 同时反演多个事件的联合概率密度函数
- **相对振幅反演** - 利用事件间的相对振幅信息
- **位置不确定性边缘化** - 通过蒙特卡洛方法处理震源位置不确定性

### 反演算法

| 算法 | 说明 |
|------|------|
| `iterate` | 迭代随机采样，运行直到达到指定采样数 |
| `time` | 时间限制随机采样，运行直到达到时间限制 |
| `mcmc` | 马尔可夫链蒙特卡洛采样 |
| `transdmcmc` | 跨维 MCMC，可在双力偶和全矩张量空间间跳转 |

## 安装

### 环境要求

- Python >= 3.10（推荐 3.13）
- NumPy >= 1.26
- SciPy >= 1.12

### 从源码安装

```bash
git clone https://github.com/mjy-888/MTfit.git
cd MTfit
pip install -e ".[dev]"
```

### 可选依赖

```bash
# MATLAB 输出格式支持
pip install -e ".[matlab]"

# 绘图功能
pip install -e ".[plotting]"

# 集群提交支持
pip install -e ".[cluster]"

# 全部安装
pip install -e ".[dev,matlab,plotting]"
```

## 快速开始

### Python API

```python
import numpy as np
from MTfit import MTfit

# 准备 P 波极性数据
data = {
    'PPolarity': {
        'Stations': {
            'Name': ['S001', 'S002', 'S003', 'S004'],
            'Azimuth': np.array([90.0, 180.0, 270.0, 0.0]),
            'TakeOffAngle': np.array([30.0, 45.0, 30.0, 60.0]),
        },
        'Measured': np.array([[1], [-1], [1], [-1]]),
        'Error': np.array([[0.05], [0.05], [0.1], [0.1]]),
    },
    'UID': 'example_event',
}

# 运行反演
MTfit(data, algorithm='time', max_time=60, parallel=False)
```

### 命令行

```bash
# P 波极性反演（时间限制 60 秒）
MTfit -d data.csv -a time -t 60

# 双力偶约束反演
MTfit -d data.csv -a iterate -x 1000000 -c

# 禁用并行（调试用）
MTfit -d data.csv -a iterate -x 100000 -l
```

## 数据格式

### Python 字典格式

```python
data = {
    'PPolarity': {
        'Stations': {
            'Name': ['STA1', 'STA2', ...],
            'Azimuth': np.array([[190], [40], ...]),
            'TakeOffAngle': np.array([[70], [40], ...]),
        },
        'Measured': np.array([[1], [-1], ...]),
        'Error': np.array([[0.01], [0.02], ...]),
    },
    'UID': 'event_id'
}
```

### 支持的数据类型

| 数据类型 | 键名 | 说明 |
|---------|------|------|
| P 波极性 | `PPolarity` | P 波初动方向（+1 上，-1 下） |
| SH 波极性 | `SHPolarity` | SH 波极性 |
| SV 波极性 | `SVPolarity` | SV 波极性 |
| P/SH 振幅比 | `P/SHRMSAmplitudeRatio` | P 波与 SH 波的均方根振幅比 |
| P/SV 振幅比 | `P/SVRMSAmplitudeRatio` | P 波与 SV 波的均方根振幅比 |
| SH/SV 振幅比 | `SH/SVRMSAmplitudeRatio` | SH 波与 SV 波的均方根振幅比 |

### CSV 文件格式

```csv
UID,Name,Azimuth,TakeOffAngle,Measured,Error
event1,S001,90.0,30.0,1,0.05
event1,S002,180.0,45.0,-1,0.05
```

## 输出格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| MATLAB | `.mat` | MATLAB 格式（v7 或 v7.3） |
| Pickle | `.pkl` | Python pickle 格式 |
| HYP | `.hyp` | NonLinLoc HYP 格式 |

## 项目结构

```
src/MTfit/
├── __init__.py                  # 包初始化
├── run.py                       # 命令行入口
├── inversion.py                 # 核心反演类
├── sampling.py                  # 采样策略
├── exceptions.py                # 自定义异常
├── algorithms/                  # 搜索算法
│   ├── base.py                  # 算法基类
│   ├── monte_carlo.py           # 蒙特卡洛采样
│   └── markov_chain_monte_carlo.py  # MCMC 算法
├── convert/                     # 矩张量参数转换
│   └── moment_tensor_conversion.py
├── probability/                 # 概率计算
│   └── probability.py
├── extensions/                  # 扩展插件系统
│   └── scatangle.py
├── utilities/                   # 工具函数
│   ├── file_io.py               # 文件读写
│   ├── argparser.py             # 命令行解析
│   └── multiprocessing_helper.py
└── tests/                       # 单元测试
```

## 测试

```bash
# 运行所有测试
pytest src/MTfit/tests/unit/ -v

# 运行特定模块测试
pytest src/MTfit/tests/unit/convert/ -v
pytest src/MTfit/tests/unit/probability/ -v
```

## 重构说明

本项目是对原始 MTfit 的 Python 3.13 现代化重构，主要改动包括：

- **NumPy 现代化**: 将所有 `np.matrix` 替换为 `np.ndarray`（500+ 处）
- **Python 3 语法**: 移除所有 Python 2 兼容代码（`cPickle`、`optparse`、`long` 类型等）
- **类型注解**: 使用 Python 3.12+ 语法为所有公共函数添加类型注解
- **构建系统**: 从 `setup.py` + `versioneer` 迁移到 `pyproject.toml`
- **依赖管理**: 从 `pkg_resources` 迁移到 `importlib.metadata`
- **现代语法**: f-string、`pathlib`、`super()` 无参数调用、`dataclass` 等

详细重构说明见 [docs/refactoring.md](docs/refactoring.md)。

## 参考文献

- Pugh, D. J., White, R. S., & Christie, P. A. F. (2016). A Bayesian method for microseismic source inversion. *Geophysical Journal International*, 206(2), 1009-1038.
- Pugh, D. J. (2015). *Bayesian Source Inversion of Microseismic Events*. PhD Thesis, University of Cambridge.

## 许可证

本代码仅供教学和非商业学术研究使用。商业应用请联系 Schlumberger 或剑桥大学。

## 致谢

原始代码作者：David J Pugh ([djpugh/MTfit](https://github.com/djpugh/MTfit))
