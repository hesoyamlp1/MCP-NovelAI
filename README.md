# NovelAI MCP Server

[![PyPI version](https://badge.fury.io/py/novelai-mcp.svg)](https://pypi.org/project/novelai-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 [novelai-python](https://github.com/LlmKira/novelai-python) SDK 的 MCP 服务器，为 AI 助手提供 NovelAI 图像生成能力。

## ✨ 功能

| 工具 | 说明 |
|------|------|
| `generate_image` | 文生图，支持多角色定位、Vibe Transfer、Precise Reference |
| `img2img` | 图生图，基于已有图片变换 |
| `suggest_tags` | Danbooru 标签搜索/自动补全 |
| `check_subscription` | 查询订阅状态和 Anlas 余额 |

### 支持模型

| 名称 | 模型 ID | 说明 |
|------|---------|------|
| `v4.5-full` | nai-diffusion-4-5-full | 最新最完整（默认） |
| `v4.5-curated` | nai-diffusion-4-5-curated | 更干净的数据集 |
| `v4-full` | nai-diffusion-4-full | V4 Full |
| `v4-curated` | nai-diffusion-4-curated-preview | V4 预览 |
| `v3` | nai-diffusion-3 | V3（老款） |

## 🚀 快速开始

### 前置条件

- Python 3.11+
- NovelAI API Key（以 `pst-` 开头，从 [novelai.net](https://novelai.net/) 获取）

### 方式一：通过 PyPI 安装（推荐）

```bash
# 使用 uvx 直接运行（无需手动安装）
NOVELAI_API_KEY=pst-xxx uvx novelai-mcp

# 或者用 pip 安装
pip install novelai-mcp
NOVELAI_API_KEY=pst-xxx novelai-mcp
```

### 方式二：从源码安装

```bash
# 克隆项目
git clone https://github.com/hesoyamlp1/MCP-NovelAI.git
cd MCP-NovelAI

# 使用 uv 安装依赖
uv sync

# 运行
NOVELAI_API_KEY=pst-xxx uv run novelai-mcp
```

### 运行选项

```bash
# Stdio 模式（默认，Claude Desktop / Antigravity / Cursor）
NOVELAI_API_KEY=pst-xxx novelai-mcp

# HTTP 模式（LobeChat / Dify）
NOVELAI_API_KEY=pst-xxx novelai-mcp --transport streamable-http --port 3000

# 或者使用 .env 文件
cp .env.example .env
# 编辑 .env 填入你的 API Key
novelai-mcp
```

### 调试

```bash
# 使用 MCP Inspector
uvx mcp dev src/novelai_mcp/server.py
```

## 🔧 接入配置

### Claude Desktop / Antigravity / Cursor

**通过 PyPI（推荐）：**

```json
{
  "mcpServers": {
    "novelai": {
      "command": "uvx",
      "args": ["novelai-mcp"],
      "env": {
        "NOVELAI_API_KEY": "pst-xxx"
      }
    }
  }
}
```

**从源码：**

```json
{
  "mcpServers": {
    "novelai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP-NovelAI", "novelai-mcp"],
      "env": {
        "NOVELAI_API_KEY": "pst-xxx"
      }
    }
  }
}
```

### LobeChat / Dify

启动 HTTP 模式后，在客户端配置：
- **URL**: `http://localhost:3000/mcp`
- **传输**: Streamable HTTP

## 🌍 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `NOVELAI_API_KEY` | ✅ | NovelAI API Key |
| `NOVELAI_SAVE_DIR` | ❌ | 图片自动保存目录 |
| `HTTPS_PROXY` | ❌ | HTTPS 代理地址 |

## 📄 许可证

MIT
