# MTfit - 贝叶斯矩张量反演程序

基于正演模型的贝叶斯矩张量反演代码，用于地震学中的震源机制解反演。

本项目是对 [djpugh/MTfit](https://github.com/djpugh/MTfit) 的 Python 3.13 现代化重构版本。原始文档参见 [djpugh.github.io/MTfit](https://djpugh.github.io/MTfit/)。

## 项目简介

MTfit 使用贝叶斯方法对地震矩张量进行反演，基于 Pugh et al. (2016) 提出的方法。程序通过蒙特卡洛采样在矩张量空间中搜索最优解，并利用贝叶斯定理边缘化各种不确定性（测量误差、位置不确定性、模型不确定性等）。

### 核心功能

- **P 波极性反演** - 利用初至 P 波极性数据约束震源机制
- **振幅比反演** - 利用 P/SH、P/SV、SH/SV 振幅比数据进行更精确的反演
- **双力偶约束反演** - 将搜索空间限制在双力偶解空间
- **全矩张量反演** - 在完整的矩张量空间中搜索
- **多事件联合反演** - 同时反演多个事件的联合概率密度函数
- **相对振幅反演** - 利用事件间的相对振幅信息
- **位置不确定性边缘化** - 通过蒙特卡洛方法处理震源位置不确定性
- **MCMC 采样** - 马尔可夫链蒙特卡洛和跨维 MCMC 采样

### 反演算法

| 算法 | 命令行参数 | 说明 |
|------|-----------|------|
| `iterate` | `-a iterate` | 迭代随机采样，运行直到达到指定采样数 |
| `time` | `-a time` | 时间限制随机采样，运行直到达到时间限制 |
| `mcmc` | `-a mcmc` | 马尔可夫链蒙特卡洛采样 |
| `transdmcmc` | `-a transdmcmc` | 跨维 MCMC，可在双力偶和全矩张量空间间跳转 |

## 安装

### 环境要求

- Python >= 3.10（推荐 3.13）
- NumPy >= 1.26
- SciPy >= 1.12
- Cython >= 3.0（可选，用于 C 扩展加速）

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/mjy-888/MTfit.git
cd MTfit

# 基本安装（可编辑模式）
pip install -e .

# 安装开发依赖（含 pytest）
pip install -e ".[dev]"

# 安装全部可选依赖
pip install -e ".[dev,matlab,plotting]"
```

### 可选依赖

```bash
# MATLAB 输出格式支持（h5py, hdf5storage）
pip install -e ".[matlab]"

# 绘图功能（matplotlib >= 3.8）
pip install -e ".[plotting]"

# 集群提交支持（pyqsub）
pip install -e ".[cluster]"
```

### 编译 Cython 扩展（可选）

Cython 扩展可显著提升概率计算和 MCMC 采样的性能，但非必需：

```bash
python setup.py build_ext --inplace
```

### 验证安装

```bash
# 检查版本
python -c "import MTfit; print(MTfit.__version__)"

# 运行单元测试
pytest src/MTfit/tests/unit/ -q --ignore=src/MTfit/tests/unit/plot/

# 运行示例测试（P 极性反演）
cd examples
python -c "from p_polarity import run; run(test=True)"
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
            'Azimuth': np.array([[90.0], [180.0], [270.0], [0.0]]),
            'TakeOffAngle': np.array([[30.0], [45.0], [30.0], [60.0]]),
        },
        'Measured': np.array([[1], [-1], [1], [-1]]),
        'Error': np.array([[0.05], [0.05], [0.1], [0.1]]),
    },
    'UID': 'example_event',
}

# 运行反演（时间限制 60 秒）
MTfit(data, algorithm='time', max_time=60, parallel=True)
```

### 使用 Inversion 类（更多控制）

```python
from MTfit.inversion import Inversion

# 创建 Inversion 对象
inv = Inversion(
    data,
    algorithm='iterate',     # 算法选择
    parallel=True,           # 并行计算
    phy_mem=1,               # 内存限制 (GB)
    dc=False,                # False=全矩张量, True=双力偶
    max_samples=1000000,     # 最大采样数
    convert=True,            # 转换为其他参数化
    inversion_options='PPolarity',  # 使用的数据类型
)

# 运行反演
inv.forward()
```

### 带位置不确定性的反演

```python
from MTfit import MTfit

MTfit(
    data,
    location_pdf_file_path='event.scatangle',  # 位置 PDF 文件
    algorithm='iterate',
    max_samples=100000,
    dc=True,
    convert=True,
    bin_scatangle=True,              # 对 scatangle 文件进行分箱
    number_location_samples=5000,    # 使用的位置采样数
)
```

### 命令行使用

```bash
# P 波极性反演（迭代采样，100 万样本）
MTfit -d data.csv -a iterate -x 1000000

# 双力偶约束反演
MTfit -d data.csv -a iterate -x 1000000 -c

# 同时运行全矩张量和双力偶反演
MTfit -d data.csv -a iterate -x 1000000 -b

# 时间限制反演（60 秒）
MTfit -d data.csv -a time -t 60

# 带位置不确定性
MTfit -d data.csv --location_pdf_file_path=event.scatangle -a iterate -x 100000 \
    --bin-scatangle --inversion-options=PPolarity

# 禁用并行（调试用）
MTfit -d data.csv -a iterate -x 100000 -l

# 查看所有选项
MTfit --help
```

### 常用命令行选项

| 选项 | 短选项 | 说明 |
|------|--------|------|
| `--data_file` | `-d` | 输入数据文件路径 |
| `--algorithm` | `-a` | 算法选择 (iterate/time/mcmc/transdmcmc) |
| `--max_samples` | `-x` | 最大采样数（iterate 算法） |
| `--max_time` | `-t` | 最大时间（time 算法，秒） |
| `--double-couple` | `-c` | 双力偶约束 |
| `--dc-mt` | `-b` | 同时运行 DC 和全 MT 反演 |
| `--parallel` / `--no-parallel` | `-l` | 启用/禁用并行 |
| `--pmem` | | 物理内存限制 (GB) |
| `--convert` | | 转换输出为多种参数化 |
| `--inversion_options` | | 指定使用的数据类型 |
| `--location_pdf_file_path` | | 位置 PDF 文件路径 |
| `--bin-scatangle` | | 对 scatangle 文件分箱 |
| `--chain_length` | | MCMC 链长度 |
| `--burn_length` | | MCMC 烧入长度 |

## 数据格式

### Python 字典格式

```python
data = {
    'PPolarity': {
        'Stations': {
            'Name': ['STA1', 'STA2', ...],
            'Azimuth': np.array([[190], [40], ...]),      # 方位角（度）
            'TakeOffAngle': np.array([[70], [40], ...]),  # 离源角（度）
        },
        'Measured': np.array([[1], [-1], ...]),   # 极性 (+1 上, -1 下)
        'Error': np.array([[0.01], [0.02], ...]), # 测量误差
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
| 极性概率 | `PPolarityProb` | P 波极性概率 |
| P/SH 振幅比 | `P/SHRMSAmplitudeRatio` | P 波与 SH 波的均方根振幅比 |
| P/SV 振幅比 | `P/SVRMSAmplitudeRatio` | P 波与 SV 波的均方根振幅比 |
| SH/SV 振幅比 | `SH/SVRMSAmplitudeRatio` | SH 波与 SV 波的均方根振幅比 |
| P 振幅 | `PAmplitude` | P 波振幅（用于相对振幅反演） |
| SH 振幅 | `SHAmplitude` | SH 波振幅 |

### CSV 文件格式

```csv
UID=event1,,,,
PPolarity,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S001,90.0,30.0,1,0.05
S002,180.0,45.0,-1,0.05
S003,270.0,30.0,1,0.1

UID=event2,,,,
PPolarity,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S001,85.0,35.0,-1,0.05
```

事件之间用空行分隔。表头行的顺序不限。

### Scatangle 文件格式

用于指定位置不确定性的射线参数：

```
Probability
StationName Azimuth TakeOffAngle
S001 90.5 30.2
S002 180.1 45.3
...

Probability
StationName Azimuth TakeOffAngle
...
```

每个段落代表一个位置采样，以概率值开头，后跟各台站的方位角和离源角。

## 示例

所有示例脚本位于 `examples/` 目录下。每个示例提供 `run(test=True)` 模式用于快速测试。

| 示例 | 文件 | 说明 |
|------|------|------|
| P 波极性 | `p_polarity.py` | 基础 P 波极性反演（iterate 算法） |
| P/SH 振幅比 | `p_sh_amplitude_ratio.py` | P 极性 + 振幅比联合反演（iterate + time） |
| 双力偶 | `double_couple.py` | 双力偶约束反演 |
| 时间限制 | `time_inversion.py` | 时间限制反演（DC + MT） |
| 合成事件 | `synthetic_event.py` | 合成数据反演（极性和振幅比） |
| Krafla 真实事件 | `krafla_event.py` | 真实数据反演，含位置不确定性和 MCMC |
| 位置不确定性 | `location_uncertainty.py` | 位置不确定性边缘化示例 |
| 相对振幅 | `relative_event.py` | 多事件联合反演（相对振幅） |
| CSV 文件 | `make_csv_file.py` | CSV 数据格式生成和解析验证 |
| MPI 并行 | `mpi.py` | MPI 并行计算示例（需要 mpi4py） |
| 命令行 | `command_line.sh` | 命令行使用示例脚本 |

### 运行示例

```bash
cd examples

# 运行 P 波极性示例（快速测试模式）
python -c "from p_polarity import run; run(test=True)"

# 运行 Krafla 真实事件示例
python -c "from krafla_event import run; run(test=True)"

# 运行合成事件示例（振幅比模式）
python -c "from synthetic_event import run; run(case='ar', test=True)"

# 运行完整示例（非测试模式，耗时较长）
python p_polarity.py
```

## 输出格式

### 支持的输出格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| MATLAB | `.mat` | MATLAB 格式（v7 或 v7.3），默认格式 |
| Pickle | `.out` | Python pickle 格式 |
| HYP | `.hyp` | NonLinLoc HYP 格式 |

### 输出文件内容

输出文件包含以下字段：

| 字段 | 说明 |
|------|------|
| `MTSpace` | 矩张量样本（6xN 数组） |
| `Probability` | 归一化概率密度 |
| `ln_pdf` | 对数概率密度（未归一化） |
| `dV` | 体积元 |
| `ln_bayesian_evidence` | 对数贝叶斯证据（如有） |
| `Dkl` | Kullback-Leibler 散度估计 |

当使用 `convert=True` 时，还包含：

| 字段 | 说明 |
|------|------|
| `g` / `d` / `k` / `h` / `s` | Tape 参数 (gamma, delta, kappa, h, sigma) |
| `S1` / `D1` / `R1` | 断层面 1 的走向/倾角/滑动角 |
| `S2` / `D2` / `R2` | 断层面 2 的走向/倾角/滑动角 |
| `u` / `v` | Hudson 参数 |

### 读取输出

```python
import scipy.io as sio

# 读取 MATLAB 格式
result = sio.loadmat('event_OutputMT.mat')
print(result.keys())

# 获取最大概率解
prob = result['Events'][0, 0]['Probability'].flatten()
mt = result['Events'][0, 0]['MTSpace']
best_idx = prob.argmax()
best_mt = mt[:, best_idx]
```

详细的理论背景和输出格式说明参见 [docs/theory_and_output.md](docs/theory_and_output.md)。

## 项目结构

```
src/MTfit/
├── __init__.py                  # 包初始化
├── run.py                       # 命令行入口（MTfit 命令）
├── inversion.py                 # 核心 Inversion 类
├── sampling.py                  # Sample/FileSample 采样存储
├── exceptions.py                # 自定义异常
├── algorithms/                  # 搜索算法
│   ├── base.py                  # 算法基类
│   ├── monte_carlo.py           # 蒙特卡洛随机采样
│   └── markov_chain_monte_carlo.py  # MCMC 算法
├── convert/                     # 矩张量参数转换
│   └── moment_tensor_conversion.py
├── probability/                 # 概率计算
│   └── probability.py           # LnPDF 类、似然函数
├── extensions/                  # 扩展插件系统
│   └── scatangle.py             # Scatangle 文件解析
├── utilities/                   # 工具函数
│   ├── file_io.py               # 文件读写（CSV、MATLAB、Pickle）
│   ├── argparser.py             # 命令行参数解析
│   └── multiprocessing_helper.py  # 多进程辅助
├── plot/                        # 绘图功能
│   ├── core.py                  # MTplot 命令入口
│   └── plot_classes.py          # 沙滩球等绘图类
└── tests/                       # 测试套件
    └── unit/                    # 单元测试
```

## 测试

```bash
# 运行所有单元测试（排除 plot 模块）
pytest src/MTfit/tests/unit/ -q --ignore=src/MTfit/tests/unit/plot/

# 运行特定模块测试
pytest src/MTfit/tests/unit/test_sampling.py -v
pytest src/MTfit/tests/unit/test_run.py -v
pytest src/MTfit/tests/unit/convert/ -v

# 运行示例集成测试
cd examples && python -m pytest test_examples.py -v --timeout=120
```

当前测试状态：207 通过，52 跳过（C 扩展未编译），57 预期失败。

## 重构说明

本项目是对原始 MTfit 的 Python 3.13 现代化重构，主要改动包括：

- **NumPy 现代化**: 将所有 `np.matrix` 替换为 `np.ndarray`（生产代码 500+ 处，示例数据 136 处）
- **Python 3 语法**: 移除所有 Python 2 兼容代码（`cPickle`、`optparse`、`long` 类型等）
- **类型注解**: 使用 Python 3.12+ 语法为公共函数添加类型注解
- **构建系统**: 从 `setup.py` + `versioneer` 迁移到 `pyproject.toml`
- **依赖管理**: 从 `pkg_resources` 迁移到 `importlib.metadata`
- **现代语法**: f-string、`pathlib`、`super()` 无参数调用、`dataclass` 等
- **Cython 扩展**: 针对 Cython 3.x 和 NumPy 2.x 重构
- **示例修复**: 修复位置不确定性和多事件反演中的空数组处理

详细重构说明见 [docs/refactoring.md](docs/refactoring.md)，变更日志见 [CHANGELOG.md](CHANGELOG.md)。

## 参考文献

- Pugh, D. J., White, R. S., & Christie, P. A. F. (2016). A Bayesian method for microseismic source inversion. *Geophysical Journal International*, 206(2), 1009-1038.
- Pugh, D. J. (2015). *Bayesian Source Inversion of Microseismic Events*. PhD Thesis, University of Cambridge.
- 原始项目文档: [djpugh.github.io/MTfit](https://djpugh.github.io/MTfit/)
- 教程: [Tutorial](https://djpugh.github.io/MTfit/tutorial.html) | [Real Data Tutorial](https://djpugh.github.io/MTfit/real-tutorial.html)

## 许可证

本代码仅供教学和非商业学术研究使用。商业应用请联系 Schlumberger 或剑桥大学。

## 致谢

原始代码作者：David J Pugh ([djpugh/MTfit](https://github.com/djpugh/MTfit))
