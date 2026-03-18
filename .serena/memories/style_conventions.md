# 代码风格与约定
- 语言：Python，字符串主要使用双引号。
- 格式化/Lint：Ruff；`line-length = 120`，启用 `E/F/B` 规则，忽略 `E711` 和 `E501`。
- 工程风格：大量 async/await；模块级单例和 manager/facade 命名较多；Pydantic/Peewee 数据模型混用。
- 代码现状：存在较多大类/大文件（>500 行）、广泛的 `except Exception`、较多透传 API 与多层 manager。后续修改应优先局部收敛复杂度，避免继续叠加抽象。