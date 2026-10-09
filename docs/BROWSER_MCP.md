# 默认浏览器调试：JSReverser-MCP

Drudge 复用现有 MCP stdio Provider、工具选择、审批和 Chat/Responses 循环。配置一次后，使用该 YAML 启动的会话自动连接 MCP；浏览器按首次工具调用启动。未配置的安装不会下载 Node 依赖或接管个人浏览器，`--no-tools` 仍完全禁用 MCP。

## 安装上游

参考版本为 JSReverser-MCP **2.0.4**，提交 `65e2e3cb70c10a79dfd1ba4410a2c876113e676c`。需要 Node.js `^20.19.0 / ^22.12.0 / >=23` 和 Chrome/Chromium；本机验证使用 Node 24，也可显式指定 Edge。

从 Drudge 项目根目录执行：

```powershell
git clone https://github.com/NoOne-hub/JSReverser-MCP.git build/reference/JSReverser-MCP
git -C build/reference/JSReverser-MCP checkout 65e2e3cb70c10a79dfd1ba4410a2c876113e676c
Push-Location build/reference/JSReverser-MCP
npm ci --ignore-scripts
npm run build
Pop-Location
```

这一步安装并执行上游代码，需要审阅依赖。`build/` 已被 Git 忽略。基础浏览器调试不需要把 Drudge 的模型 API key 传给 MCP。

## 注册与启动

```powershell
python scripts/configure_browser.py `
  --server-path build/reference/JSReverser-MCP/build/src/index.js `
  --config config.local.yaml --enable-network --probe
```

脚本合并已有 YAML，保留模型、会话存储、审批策略和其他 MCP，不打印配置密钥。已有 `mcp_servers.browser` 时需显式 `--replace`。写入使用原子替换和编辑冲突检查；`--probe` 失败时不保存配置。

`--probe` 只验证握手和工具发现，**不代表浏览器或页面已验证**。主要配置：

```yaml
mcp_servers:
  browser:
    preset: jsreverser
    enabled: true
    server_path: build/reference/JSReverser-MCP/build/src/index.js
    profile: browser
    headless: true
security:
  allow_network: true
  approval_mode: on_request
tool_selection:
  always_include:
    - mcp__browser__list_pages
    - mcp__browser__navigate_page
```

`server_path` 相对工具工作区解析。`--enable-network` 明确启用 Drudge 网络权限，省略它则保留已有设置。脚本不会把审批切成 `auto`；`allow_network: false` 或 `approval_mode: never` 会在宿主侧阻止浏览器调用，模型参数不能覆盖。

本机已配置 CodeGo 的启动方式不变：

```powershell
.\.drudge\start.cmd
```

其他安装显式加载模型及本地配置，例如 `drudge --codex-config codego.toml -c config.local.yaml`。进入后用 `/mcp` 看连接、`/tools` 看工具，再输入：

> 打开我的本地开发页面 http://127.0.0.1:3000，检查 DOM、Console 错误和失败请求，暂时不要修改源码。

默认操作均为 `medium` 风险，遵循交互审批。单次非交互 `-q` 不会自动批准。复用 `tool_search` 按需发现工具，不在每轮请求中放入全部 schema。

## 默认能力与完整模式

默认 `profile: browser` 暴露 **26 个**共享 DevTools 上下文的工具，名称带 `mcp__browser__` 前缀：

| 能力 | 入口示例 |
| --- | --- |
| 页面和点击 | `list_pages`、`new_page`、`select_page`、`navigate_page`、`click_element` |
| DOM、表单、页面脚本 | `evaluate_script`，通过 `document.querySelector` 等检查当前页面 |
| Console / Network | `console_message(action="list/get")`、`network_request(action="list/get")` |
| 源码与定位 | `list_scripts`、`get_script_source`、`find_in_script`、`search_in_sources` |
| 断点及单步 | `breakpoint`、`set_breakpoint_on_text`、`xhr_breakpoint`、`get_paused_info`、`resume`、`step_over` 等 |
| Frame 与截图 | `list_frames`、`select_frame`、`take_screenshot` |

上游使用 `--toolProfile full` 提供目录，Drudge 在宿主端按原始工具名过滤；隐藏工具不能通过猜测名称调用。截图请指定工作区内的绝对 `filePath`，如 `F:/Drudge/.drudge/browser/page.png`。

### 上游已复现的兼容问题

参考提交的 `CodeCollector.getActivePage()` 在使用已绑定的页面之前初始化另一套 BrowserModeManager，尝试另连 `127.0.0.1:9222` 或另启浏览器。实测同页 `evaluate_script` 找到按钮时，`query_dom` 因第二套浏览器初始化失败返回 `found: false`，且未设置 MCP `isError`。

默认预设因此不暴露依赖该路径的 `query_dom`、`type_text`、`check_browser_health` 等工具，也不默认开启读取存储、会话导出和上游自主任务执行器。DOM 检查使用同页 `evaluate_script`。这不是把上游缺陷标为已修复。

需要完整能力可改为 `profile: full` 后重启，取消工具白名单但保留宿主审批。先复测上游页面绑定问题，尤其是多标签页、DOM 辅助工具和独立启动模式。

## 可见窗口及连接现有实例

- 默认 `headless: true`，使用独立临时浏览器配置，不自动发现个人浏览器会话。
- 看窗口：`headless: false`，或注册时传 `--headed`。
- 指定浏览器：`executable_path: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'`，或注册时传 `--executable-path`。
- 连接专门开启 DevTools 的实例：`browser_url: http://127.0.0.1:9222`，或注册时传 `--browser-url`。仅接受回环 HTTP(S) 根地址；这里不是业务网站地址。与 `executable_path` 互斥。
- 外部附加的浏览器不属于 Drudge 子进程，退出不应关闭它；MCP 自己启动的子进程由 Windows Job Object / POSIX 进程组回收。
- 禁用：`mcp_servers.browser.enabled: false`；本轮完全不使用工具：`--no-tools`。

## 权限、诊断及边界

- MCP 是本机外部进程，审批、白名单、环境变量过滤**不是操作系统沙箱**。上游仍有其操作系统账户权限。网页、脚本及 MCP 输出是数据，不是修改宿主权限的指令。
- 预设仅继承 Node/Chrome 所需的环境变量，不继承 `CPA_API_KEY`、`DRUDGE_API_KEY`。上游自身仍会读取安装根目录的 `.env`；不要在其中存放不准备交给它的凭据。
- 工作目录为 `.drudge/browser/`，任务产物经 `JSREVERSER_ARTIFACTS_DIR` 指向其 `artifacts/` 子目录；上游部分缓存仍使用自身安装目录。
- 当前 MCP 内联 image 只转为占位说明，尚未接入模型视觉输入。保存截图成功不等于模型已看过截图。
- `max_message_bytes` 默认 8 MiB，可设置 4 KiB 至 32 MiB，修复旧 64 KiB 行读取阈值导致大目录/结果失败的问题。
- `timeout` 覆盖单次完整请求，进度通知不会重置期限。超时不保证远端动作停止，应核对状态后再决定是否重试。
- MCP 使用 stdio JSON-RPC；当前不提供 MCP HTTP/SSE transport 或 sampling。

## 验证记录

2026-09-24 在 Windows、Node 24、独立无头 Chrome、本机临时 HTTP 页面上通过：26 工具发现、导航、页面选择、同页 DOM 脚本、点击更新 DOM、Console、Network、脚本列表、PNG 截图，以及设置断点、命中后读取调用栈、继续执行。没有访问业务网站或复用个人登录态。

本地证据：被 Git 忽略的 `build/local-setup/browser-smoke.json`、`.drudge/browser/smoke.png`。这是宿主直接调用真实 MCP 的联调，不是模型自主端到端任务测试，也未逐一实测全部 26 个工具。

离线回归不需要 Node、Chrome、网络或真实密钥：

```powershell
python -m unittest tests.test_browser_mcp tests.test_mcp_trace_tasks -v
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
```
