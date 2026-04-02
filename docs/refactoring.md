# MTfit 重构技术文档

## 重构概述

MTfit 从 Python 2.7/3.5/3.6 兼容的代码库重构为纯 Python 3.13 代码。重构按照依赖关系自底向上进行，每个模块重构后立即运行测试验证。

## 重构顺序（按依赖关系）

| 阶段 | 模块 | 主要改动 | 测试结果 |
|------|------|---------|---------|
| Phase 0 | 构建系统 | pyproject.toml, _version.py, conftest.py, exceptions.py | 基础设施就绪 |
| Phase 1 | convert/moment_tensor_conversion.py | 24 处 np.matrix→ndarray, 类型注解 | 53 通过 |
| Phase 2 | probability/probability.py | LnPDF 类迁移, 移除 `__getslice__` | 33 通过 |
| Phase 3 | sampling.py + algorithms/ | 移除 long=int, 23+ super() 现代化 | 55 通过 |
| Phase 4 | utilities/ | 移除 optparse/cPickle/unicode shim | 23 通过 |
| Phase 5 | extensions/scatangle.py | 移除 argparse try/except, pathlib | 1 通过 |
| Phase 6 | inversion.py | 29 处 np.matrix, 5 个类现代化 | 12 通过 |
| Phase 7 | run.py + __init__.py | pathlib, importlib.metadata | 181 总计通过 |

## np.matrix → np.ndarray 迁移策略

### 为什么要迁移

`np.matrix` 自 NumPy 1.15 起已弃用，在 NumPy 2.x 中将被移除。它存在以下问题：
- 强制所有数组为 2D，不够灵活
- `*` 运算符行为不同（矩阵乘法 vs 逐元素乘法）
- `.T` 在 1D 数组上的行为不同
- 与现代 NumPy API 不兼容

### 迁移规则

1. **形状保持**: `np.matrix` 始终是 2D，而 `np.ndarray` 可以是 1D。需要在关键位置使用 `np.atleast_2d()` 或 `.reshape()` 保持 2D。

2. **乘法运算符**: `np.matrix * np.matrix` = 矩阵乘法，`np.ndarray * np.ndarray` = 逐元素乘法。扫描所有 `*` 运算符，将矩阵乘法改为 `@` 运算符。

3. **转置**: `np.matrix` 上的 `.T` 始终返回 2D。1D `np.ndarray` 上的 `.T` 是空操作。需要使用 `.reshape(-1, 1)` 创建列向量。

4. **输入容错**: 所有公共函数在入口处添加 `np.asarray()` 转换，确保即使传入 `np.matrix` 也能正确处理。

5. **标量提取**: NumPy 2.x 中 `float()` 对 1 元素数组不再有效。使用 `.item()` 或 `.flat[0]` 代替。

### 关键改动示例

```python
# 之前 (np.matrix)
MT6 = np.matrix([[MT33[0,0]], [MT33[1,1]], [MT33[2,2]],
                  [np.sqrt(2)*MT33[0,1]], [np.sqrt(2)*MT33[0,2]],
                  [np.sqrt(2)*MT33[1,2]]])

# 之后 (np.ndarray)
MT6 = np.array([[MT33[0,0]], [MT33[1,1]], [MT33[2,2]],
                 [np.sqrt(2)*MT33[0,1]], [np.sqrt(2)*MT33[0,2]],
                 [np.sqrt(2)*MT33[1,2]]])
```

```python
# 之前: LnPDF 内部存储
self._ln_pdf = np.matrix([])

# 之后: 始终保持 2D ndarray
self._ln_pdf = np.empty((1, 0))
```

## 构建系统迁移

### 之前
- `setup.py` (165 行) + `versioneer.py` (68,587 字节)
- 通过 git tag 自动生成版本号
- Travis CI + AppVeyor 进行 CI/CD

### 之后
- `pyproject.toml` (93 行)
- `importlib.metadata` 获取版本号（18 行 `_version.py`）
- GitHub Actions CI/CD（待配置）

## 删除的文件

| 文件 | 原因 |
|------|------|
| `setup.py` | 被 `pyproject.toml` 替代 |
| `versioneer.py` | 被 `importlib.metadata` 替代 |
| `.travis.yml` | 被 GitHub Actions 替代 |
| `appveyor.yml` | 被 GitHub Actions 替代 |
| `tox.ini` | 被 pyproject.toml 中的 pytest 配置替代 |
| `Pipfile` / `Pipfile.lock` | 被 pyproject.toml 替代 |

## 统计

- **生产代码中 np.matrix**: 0（排除 plot/）
- **示例代码中 np.matrix**: 0（example_data.py 136 处已转换）
- **cPickle 导入**: 0
- **optparse 导入**: 0
- **super(Cls, self) 模式**: 0
- **sys.version_info 兼容检查**: 0
- **总测试通过**: 207（单元测试），10 个示例全部通过
- **代码净减少**: ~3000+ 行

---

## Cython 扩展重构（Phase 2）

### 概述

4 个 Cython 扩展文件（共 ~3845 行）针对 Cython 3.x 和 NumPy 2.x 进行了现代化重构。

### 通用改动（所有 .pyx 文件）

| 改动 | 原因 |
|------|------|
| 移除 `from cython.view cimport array as cvarray` | Cython 3.x 弃用 |
| 移除 `from cpython cimport bool` | Cython 3.x 弃用，改用 Python 内置 bool |
| 添加 `np.import_array()` | NumPy 2.x C-API 初始化要求 |
| `for i from 0<=i<n:` → `for i in range(n):` | C 风格循环在 Cython 3.x 弃用 |
| 清理注释掉的 debug print 语句 | 代码清洁 |
| 将嵌入的 TestCase 移到独立测试文件 | Cython 3.x 不支持 .pyx 中的 Python 类包含 cdef |

### 各文件改动详情

#### cprobability.pyx（2172 → 2137 行）
- 替换 127 处弃用的 C 风格 for 循环
- 清理 18 处 debug print 注释
- 保留 Windows 平台 erf() 条件编译
- 无 np.matrix 使用

#### cmoment_tensor_conversion.pyx（726 → 511 行）
- **修复 np.matrix 使用**（2处）→ `np.atleast_2d(np.asarray(...))`
- 移除嵌入的 `cMomentTensorConvertTestCase`（~215 行）→ 新测试文件
- 添加 `ctypedef double DTYPE_t`（之前仅在 .pxd 中定义）

#### cmarkov_chain_monte_carlo.pyx（866 → 603 行）
- 移除嵌入的 `_cmarkov_chain_monte_carlo_TestCase`（~260 行）→ 新测试文件
- 修复 9 处 `np.random.randn(1)` → `.item()`（NumPy 2.x 兼容）
- 保留测试辅助函数（`_acceptance_test_fn` 等）

#### cscatangle.pyx（81 → 65 行）
- 统一使用 `HUGE_VAL` 替代平台条件编译
- 最小改动，代码最简单

### 构建系统

新增最小化 `setup.py` 专门用于 Cython 扩展编译：
```bash
# 编译 C 扩展（可选，不影响纯 Python 功能）
python setup.py build_ext --inplace
```

### 新增测试文件
- `tests/unit/convert/test_cmoment_tensor_conversion.py` — 从 .pyx 移出的 C 扩展测试
- `tests/unit/algorithms/test_cmarkov_chain_monte_carlo.py` — 从 .pyx 移出的 MCMC 扩展测试

### 仓库清理

删除了 19 个无关文件：
- 旧 CI 脚本（ci/ 目录，.travis.yml，appveyor.yml）
- 旧构建配置（setup.cfg，MANIFEST.in，requirements.txt，tox.ini）
- Vagrant 开发环境
- 旧 README.rst
- 测试输出文件（.mat，.out）

---

## Phase 3: 示例复现修复

### 问题描述

重构后部分示例无法运行，涉及位置不确定性（krafla_event、location_uncertainty）和多事件反演（relative_event）。

### 根因

1. **空数组维度不匹配**: `inversion.py` 中 9 处返回 `np.array([])` (shape=(0,), 1D)，而下游 `sampling.py` 假设所有数组为 2D（np.matrix 遗留假设）
2. **Python 3 字典迭代**: `sampling._convert()` 在迭代 `dict.keys()` 时修改字典，Python 2 中 keys() 返回列表不受影响，Python 3 返回视图导致 RuntimeError
3. **空 PDF 处理**: `ln_normalise()` 对空数组调用 `.max()` 引发 ValueError
4. **示例数据格式**: `example_data.py` 仍使用 `np.matrix`，与 CSV 解析器返回的 `np.ndarray` 不兼容

### 修复

| 文件 | 改动 |
|------|------|
| `inversion.py` | 9 处 `np.array([])` → `np.empty((n, 0))` |
| `sampling.py` | 空数组早期返回 (2处) + `list(dict.keys())` (1处) |
| `probability.py` | `ln_normalise` 空数组检查 (1处) |
| `example_data.py` | `from numpy import array as matrix` (136 处调用) |
| `make_csv_file.py` | 类型比较兼容 ndarray/matrix (1处) |
