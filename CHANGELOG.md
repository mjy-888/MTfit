# 变更日志

## v2.0.0 - Python 3.13 现代化重构

### 构建系统
- 用 `pyproject.toml` 替换 `setup.py` + `versioneer`
- 用 `importlib.metadata` 替换 `pkg_resources` 和 `versioneer`
- 要求 Python >= 3.10，推荐 Python 3.13
- 依赖升级：NumPy >= 1.26, SciPy >= 1.12, Cython >= 3.0

### NumPy 现代化
- **将所有 `np.matrix` 替换为 `np.ndarray`**（生产代码中 500+ 处，测试代码中 300+ 处）
- 矩阵乘法 `*` 替换为 `@` 运算符
- 修复 NumPy 2.x 中 `float()` 对 1 元素数组不再有效的问题，改用 `.item()` 或 `.flat[0]`
- `np.matrix` 的自动 2D 行为改为显式使用 `np.atleast_2d()`

### Python 2 兼容代码移除
- 移除 `try: import cPickle as pickle`（5 个文件）
- 移除 `if sys.version_info.major >= 3: long = int`
- 移除 43+ 处 `sys.version_info` 版本检查（34 个文件）
- 移除 `convert_keys_to_unicode` / `convert_keys_from_unicode`（Python 2 shim）
- 移除所有 `optparse` 代码（~470 行），仅保留 `argparse`
- 移除 `types.MethodType` / `.func_name` vs `.__name__` 检查

### 现代 Python 语法
- `class Foo(object):` → `class Foo:`（26+ 个文件）
- `super(ClassName, self).__init__()` → `super().__init__()`（43+ 处）
- `.format()` → f-string
- `os.path.*` → `pathlib.Path`
- 类型注解使用 Python 3.12+ 语法（`list[str]`、`X | Y`）
- 输出任务类转换为 `@dataclass`

### 新增
- `src/MTfit/exceptions.py` - 自定义异常类层次结构
- `src/MTfit/tests/conftest.py` - pytest 共享 fixture
- `pyproject.toml` - 现代化 Python 包配置

### 测试
- 测试结果：181 通过，45 跳过（C 扩展未编译），55 预期失败
- 所有测试中的 `np.matrix` 替换为 `np.array`
- 测试中直接使用 `from unittest import mock`（移除版本检查）
- 测试中直接使用 `tempfile.TemporaryDirectory()`（移除 Python 2 分支）
