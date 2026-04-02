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
- **cPickle 导入**: 0
- **optparse 导入**: 0
- **super(Cls, self) 模式**: 0
- **sys.version_info 兼容检查**: 0
- **总测试通过**: 181
- **代码净减少**: ~1825 行
