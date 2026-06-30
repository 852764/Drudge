# Drudge

Drudge 是一个轻量级终端 AI Agent 原型，目标是逐步演进成类似 Drudge / Codex 的本地开发助手。

## 当前能力

- OpenAI-compatible Chat Completions 客户端
- Responses API 与函数工具调用适配
- 终端、文件、Web 三类工具注册
- 单次查询与交互式 CLI
- YAML 配置文件与环境变量配置
- 不可变工具权限上下文和显式 Agent 运行状态

## 安装

```bash
pip install -e .
```

## 配置

默认读取环境变量：

```bash
set OPENAI_API_KEY=your_api_key
set DRUDGE_MODEL=gpt-4o-mini
set DRUDGE_BASE_URL=https://api.openai.com/v1
```

也可以传入 YAML 配置：

```yaml
model:
  name: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key: your_api_key
toolsets:
  - terminal
  - file
  - web

agent:
  refusal_review_enabled: true
  refusal_review_notice: "[Drudge] 检测到模型可能拒绝了请求，正在进行安全二次处理..."
  # Optional: override the second-pass review model/provider.
  # refusal_review_model:
  #   name: gpt-4o-mini
```

## 使用

```bash
drudge --version
drudge --help
drudge -q "列出当前目录文件"
drudge -c config.yaml -m gpt-4o-mini
drudge --codex-config -q "检查当前项目"
drudge --codex-config C:\path\to\config.toml
```

开发时也可以直接运行：

```bash
python main.py --version
python main.py --help
```

## 测试

测试完全离线，不需要 API Key：

```bash
python -m unittest discover -s tests -v
```

## 开发文档

- [第一周实施说明](docs/WEEK1_IMPLEMENTATION.md)
- [使用 Codex 配置](docs/CODEX_CONFIG.md)
- [Drudge Codex OAuth](docs/CODEX_OAUTH.md)
- [Priority 1-3 Implementation Notes](docs/PRIORITY_1_2_3.md)
