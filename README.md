# Drudge

Drudge 是一个轻量级终端 AI Agent 原型，目标是逐步演进成类似 Hermes / Codex 的本地开发助手。

## 当前能力

- OpenAI-compatible Chat Completions 客户端
- 终端、文件、Web 三类工具注册
- 单次查询与交互式 CLI
- YAML 配置文件与环境变量配置

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
```

## 使用

```bash
drudge --version
drudge --help
drudge -q "列出当前目录文件"
drudge -c config.yaml -m gpt-4o-mini
```

开发时也可以直接运行：

```bash
python main.py --version
python main.py --help
```
