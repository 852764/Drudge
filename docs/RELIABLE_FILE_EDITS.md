# 可靠文件编辑与撤销预览

本轮重点是保护开发者正在编辑的文件，而不是增加更多工具名。保持 Python 3.10+、Chat Completions / Responses 和旧工具参数兼容。

## 内容指纹：避免基于过期内容编辑

`read_file` 新增 `sha256`（同时出现在标准结果的 `metadata` 中），覆盖**整个文件的原始字节**，不受 `offset` / `limit` 分页影响。

`write_file`、`patch`、`apply_patch` 新增可选字符串参数 `expected_sha256`：

```json
{
  "path": "src/example.py",
  "old_string": "return 1",
  "new_string": "return 2",
  "expected_sha256": "<read_file 返回的 64 位 SHA-256>"
}
```

创建新文件时用 `write_file` 配合 `"expected_sha256": "missing"`，即使已有文件为空也会报告冲突。省略参数保留旧调用方式；系统提示词指导模型主动使用指纹，并在冲突时重新读取、审查差异，不通过移除指纹重试。

冲突通过标准工具协议返回，两个 API 的工具事务均能继续：

```json
{
  "ok": false,
  "error": "File changed since the snapshot: ...",
  "metadata": {
    "conflict": true,
    "expected_sha256": "...",
    "actual_sha256": "..."
  }
}
```

文件缺失时对应指纹为 `null`。成功编辑返回新的 `sha256`、`before_sha256`、`changed`、`diff_summary` 和 `checkpoint_created`。

## 原子写入与字节精度

- 同目录临时文件写入、flush / fsync，提交前再次比对源文件，再由 `os.replace` 替换。
- 临时写入或替换失败时保留原文件并清理临时文件；已有文件的权限位保持不变。
- UTF-8 字节写入不经过操作系统换行转换。整文件写入遵循传入文本；局部编辑保留未修改区域、BOM 和文件末尾状态。
- 对统一 CRLF 文件，局部编辑会把模型传入的 LF 多行文本适配为 CRLF；混合换行文件按原文匹配。
- 空 `old_string` 报错；无变化的编辑不重写文件、不增加检查点。
- 文件保存成功但数据库检查点失败时，返回 `ok: true`、`checkpoint_created: false` 和 `warnings`，明确文件已保存，避免模型重复修改。

这是乐观冲突检测与单文件原子替换，不是对任意外部进程的排他锁：最后一次比对到替换之间仍有竞争窗口。也不承诺保留 inode、硬链接关系、ACL 或扩展属性。

## 撤销前预览与冲突检测

```text
/changes
/undo --dry-run
/undo
```

预览显示最近一项可撤销变更的路径、恢复 / 删除动作和反向 diff 摘要，不改变文件及 SQLite 撤销标记。实际 `/undo` 会重新校验：

- 当前文件必须与该检查点的编辑后内容逐字节一致；用户后续编辑、删除或新建文件造成不一致时保留现状。
- 新建文件的撤销是删除，原有空文件的撤销是恢复为空，两者不混淆。
- 使用当前宿主 `ToolContext`：工作区、凭据路径保护、文件工具开关和 `approval_mode=never` 均生效。
- `/undo` 是用户显式发起的本地命令，`on_request` 下视作本次撤销的确认；`never` 下仍可只读预览。
- 文件恢复失败时不标记已撤销；数据库写事务防止两个 Drudge 撤销消费者重复应用同一检查点。

SQLite 和文件系统不是一个跨资源事务。进程在文件恢复与数据库提交之间崩溃时，需检查实际文件状态；后续冲突检查不会静默重复覆盖。

## 旧数据库迁移

`file_revisions` 幂等新增 `snapshot_version`、`before_sha256`、`after_sha256`。新记录为版本 1，使用原始 UTF-8 文本和字节指纹；原有记录、ID、撤销状态与时间戳保留。

旧记录为版本 0，不伪造字节指纹。撤销仅在当前文件与旧文本的 UTF-8 表示完全相同时继续；旧版 Windows 换行转换导致不一致时报告冲突，不猜测历史字节。

## 上下文和路径保护

凭据路径检查移到宿主路径解析中，覆盖搜索发现的文件和撤销。AGENTS.md 自定义文件名、Skill 文档与引用也排除凭据路径及解析到这些路径的别名；上下文读取按字符预算进行。

## 离线回归

新增测试覆盖：两种 API 的带指纹工具调用、过期内容、空文件、换行与 BOM、写入异常、检查点失败、撤销预览 / 冲突 / 宿主权限 / 并发消费、迁移幂等性，以及上下文路径保护。

```powershell
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
```
