# 仓库搜索与上下文发现（0.2.0b3）

`search_files` 与仓库概览现在共用文件发现规则，去除了过去“先枚举所有文件、再只查前 200 个”的静默漏查。

## 查询与定位

```json
{
  "pattern": "class RepositoryWalker",
  "path": "tools",
  "file_glob": "**/*.py",
  "literal": true,
  "case_sensitive": true,
  "limit": 50
}
```

- 默认仍是大小写不敏感的 Python 正则；`literal: true` 按字面量匹配。
- `limit` 为 1–1000，默认 50，每行最多返回一个匹配。
- `*.py` 在任意深度匹配文件名；`src/**/*.py` 按搜索根相对路径匹配，`**` 可代表零层或多层。文件 glob 大小写敏感。
- 返回绝对路径、从 1 开始的 `line`、`column`，以及围绕匹配位置的最多 200 字符片段。`content_start_column` 标明片段起始列；列号按 Unicode 字符计，不是 UTF-8 字节数。
- 恰好得到 `limit` 个结果且到达范围末尾时，不再误报截断；只有发现额外匹配或遇到扫描限制才报告不完整。

## 文件发现规则

使用 `pathspec` 解释工作区内根及嵌套 `.gitignore`，支持否定规则。搜索子目录时继承宿主工作区内的祖先规则。已经被忽略的父目录会被剪枝，其子目录规则不会重新启用它。

这是文件系统发现，不读取 Git 索引、全局排除文件或 `.git/info/exclude`；已被 Git 跟踪的文件也仍受这些发现规则影响。忽略规则每次操作重新加载，不依赖陈旧索引。

默认不遍历普通隐藏路径；`include_hidden: true` 可以包括它们。隐藏判断不再检查工作区外的祖先目录。以下仍始终从目录发现中排除：

- `.git`、`.drudge`、`.codex` 等私密/运行时目录；
- 虚拟环境、`node_modules`、缓存、`build`、`dist`；
- 数据库、字节码、`.env` 和 `.env.*`（`.env.example` 使用普通隐藏规则）；
- 符号链接、Windows junction/reparse points 和特殊文件。

每层先处理普通文件，再处理子目录，分别稳定排序。过大的目录整体跳过并报告 `entry_limit`，不基于文件系统返回顺序选择任意前缀。

**显式指定单个文件**会绕过发现过滤及 glob，但不绕过宿主路径权限、凭据保护或读取预算。需要查看某个已知的生成日志时可采用这种方式。

## 完整性不等于存储完整性

保留原有 `matches`、`total`、`truncated` 字段，新增：

- `complete`：本次查询的合格 UTF-8 文本范围是否查完。它不是整个磁盘的快照，也不包含被策略过滤的文件；
- `incomplete_reasons`：如 `match_limit`、`file_limit`、`byte_limit`、`file_size_limit`、`entry_limit`、`ignore_unreadable`；
- `files_scanned`：进入读取阶段的候选数，包括随后判断为二进制或非 UTF-8 的文件；
- `bytes_read`、`entries_seen`、`skipped`：读取量、发现量与跳过计数；
- `scope`、`limits`：实际查询范围及宿主预算。

读取故障或预算耗尽会令 `complete=false`、`truncated=true`，即使没有匹配也不代表目标不存在。应检查原因并缩小 `path` 或 `file_glob`。二进制/NUL 和非 UTF-8 内容被排除在文本范围外，并单独计数。

结果过大时，工具预览仍保留搜索 `complete`；默认预算内保留最多 8 条、每条最多 80 字符的不完整原因。特别小的预览预算可能省略原因列表，完整记录通过 output receipt 读取。**`metadata.output_ref.complete=true` 只表示结果封装完整保存，与搜索是否完整是两回事。**

## 宿主预算

可在 Drudge YAML 的 `security` 中配置，模型工具参数不能修改：

```yaml
security:
  search_max_files: 10000
  search_max_file_bytes: 2097152
  search_max_total_bytes: 33554432
  search_max_entries: 50000
```

这些值必须是正整数。单个 ignore 文件最多 64 KiB、2048 行、每行 4096 字符；整个发现过程最多读取 1 MiB ignore 数据。无法正确加载某目录的忽略策略时，跳过该目录并明确报告，不带着缺失策略继续扫描。忽略文件同样在打开前经过宿主路径检查。

上述限制约束 I/O 和发现规模，**没有实现正则 CPU 超时**。复杂回溯正则仍可能耗时；优先用 `literal` 查询或简单表达式。并发修改文件也不会形成原子仓库快照。

仓库概览默认最多 80 个文件、深度 3；它只输出实际列出文件的父目录，避免空目录撑大上下文。概览受到限制时会附带提示。

## 本轮验证

离线测试覆盖 200 文件之后的匹配、隐藏祖先、嵌套规则与继承、否定/剪枝、预算、精确截断、Unicode/BOM、权限和 ignore 别名、两种模型 API 的工具调用，以及大结果完整性保留。

一次真实 CodeGo `gpt-6-astra` / `xhigh` / Responses HTTP/SSE 试用中，在隐藏祖先目录下新建 301 个源码文件，并放入一个被 `.gitignore` 排除的同名干扰项。Agent 一次 `search_files` 扫描 301 个源码，只找到排序靠后的正确目标，返回 `complete=true`；2 轮完成、19.01 秒、7,190 reported tokens，文件哈希未改变。报告保存在被忽略的 `build/dogfood/search-_4qsbdrt/report.json`。

这是单次功能验收，不是稳定性能基准或竞品排名；未测试 WebSocket。临时工作区、会话数据库和认证数据不进入发布源码。
