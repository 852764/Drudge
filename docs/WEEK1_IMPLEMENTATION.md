# Drudge 第一周实施说明

## 目标与结果

本阶段先稳定执行内核，再扩展 MCP 和 Skills。已完成：

1. 移除源码中的默认 API Key，增加配置脱敏输出和仓库忽略规则。
2. 引入不可变 `ToolContext`，把模型参数与宿主权限彻底分离。
3. 在工具 dispatch 层增加 toolset 白名单、未知参数、必填参数和类型校验。
4. 将 Agent loop 改为显式运行状态机。
5. 补全 Responses API 的工具 schema、函数调用和函数结果转换。
6. 建立不依赖网络和真实模型的 fake LLM 测试基线。

外部遗留事项：曾出现在工作区中的旧 API Key 必须到对应供应商控制台立即吊销。代码只能停止继续使用或输出它，不能代替供应商完成吊销。

## 安全模型

### ToolContext

`tools/context.py` 定义宿主创建的不可变上下文：

- `workspace`
- `enabled_toolsets`
- `allow_outside_workspace`
- `allow_terminal`
- `allow_network`

这些字段不属于模型可见的工具 schema。Registry 会先拒绝 schema 外参数，再由宿主把上下文注入 handler。因此模型即使生成 `allow_outside_workspace=true`，也只会得到 blocked 错误，无法覆盖宿主策略。

执行路径如下：

```text
model tool arguments
        |
        v
schema/required/type validation
        |
        v
enabled toolset check
        |
        v
host injects immutable ToolContext
        |
        v
tool handler
```

文件路径在访问前通过 workspace 解析；终端工作目录使用同一规则。`allow_terminal` 和 `allow_network` 在 handler 内再次检查。

### 当前边界

第一周完成的是不可绕过的配置边界，不是完整 OS 沙箱。以下内容安排到下一阶段：

- 写文件、执行命令和外部网络请求的交互式审批
- terminal 命令级策略和子进程取消
- Web SSRF、私网地址和域名策略
- OS 级进程、文件系统和网络隔离

字符串危险命令黑名单只能作为补充，不能视为安全边界。

## Agent 状态机

`agent/state.py` 定义以下状态：

```text
idle
  -> waiting_for_model
       -> executing_tools -> waiting_for_model
       -> completed
       -> failed
  -> max_turns
```

每次转换产生 `RunEvent`，记录 request turn、总 turn、工具数量、结束原因或错误。调用方可通过 `agent.get_run_state()` 获取最终状态和事件轨迹。

循环行为变更：

- 有工具调用时，优先执行工具，不再把中间说明拼入最终答案。
- 工具参数 JSON 损坏时，将结构化错误返回模型，使其有机会修复。
- 模型异常、空响应和最大轮数分别进入明确状态。
- 同步工具通过 worker thread 执行，避免阻塞 async Agent loop。

## Responses API 适配

Responses API 使用扁平函数工具格式，而 Chat Completions 使用嵌套 `function` 格式。Provider adapter 负责双向转换，Agent 继续使用统一的内部 Chat 风格消息。

已支持：

- Chat tool schema 转为 Responses function tool
- `output[].type == function_call` 转为内部 tool call
- 工具结果转为 `function_call_output`
- 使用 `call_id` 关联调用与结果
- 保留 Responses 原始 output items，下一轮原样回传 reasoning/function items
- `completed` 映射为 `stop`
- `incomplete` 映射为 `length`

协议实现依据 OpenAI 官方 [Function calling guide](https://developers.openai.com/api/docs/guides/function-calling) 和 [Responses create reference](https://developers.openai.com/api/reference/resources/responses/methods/create)。

## 测试基线

运行：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- API Key 和嵌套 token 脱敏
- JSON Schema 禁止额外属性
- 模型不能覆盖 ToolContext
- workspace 路径穿越被阻止
- disabled toolset 无法 dispatch
- terminal 权限由上下文强制执行
- fake LLM 完整工具循环
- 中间文本不会污染最终答案
- provider failure、损坏参数和 max turns 状态
- Responses 扁平工具 schema
- `function_call` / `function_call_output` 转换
- Responses `completed` 正确结束

所有测试使用临时目录和 fake provider，不调用真实 API。

## 关键文件

- `tools/context.py`：宿主权限上下文
- `tools/registry.py`：工具参数和执行权限校验
- `agent/state.py`：运行状态与事件
- `agent/drudge_agent.py`：状态机驱动的工具循环
- `agent/llm.py`：Chat Completions / Responses adapter
- `tests/fakes.py`：离线模型 fake
- `tests/test_tools_security.py`：权限回归测试
- `tests/test_agent_loop.py`：Agent loop 集成测试
- `tests/test_responses_adapter.py`：Responses 协议测试

## 下一阶段入口

按风险和依赖顺序继续：

1. approval UI 和风险分级。
2. streaming、用户取消和子进程取消。
3. session resume/checkpoint。
4. 保持完整 tool transaction 的上下文压缩。
5. 项目指令文件加载。
6. 在稳定的 ToolProvider 接口上增加 MCP。
