# AI Tool Assistant

一个基于 **Python + LangChain + LLM + Tool Calling** 的简单 AI 工具助手。

它让大语言模型（LLM）根据用户的问题，**自主判断**是否需要调用工具、调用哪个工具、给工具传什么参数，并在工具执行完成后结合结果生成最终回答。

## 功能

内置三个工具，全部使用免费公开 API（无需额外注册）：

| 工具 | 功能 | 底层 API |
|------|------|----------|
| `get_exchange_rate` | 查实时汇率 | Frankfurter |
| `get_weather` | 查城市当前天气 | Open-Meteo |
| `get_random_quote` | 获取随机英文名言 | DummyJSON |

## 技术栈

- Python 3.11
- LangChain（`langchain_core.tools` + `model.bind_tools`，现代 API）
- langchain-openai（连接 DeepSeek）
- FastAPI + uvicorn
- httpx / Pydantic / python-dotenv

## 项目结构

```
.
├── main.py            # FastAPI 入口 + Tool Calling 主流程
├── tools.py           # 三个工具的定义
├── config.py          # 加载配置、初始化 LLM、绑定工具
├── static/index.html  # 前端页面
├── .env.example       # API Key 模板
├── requirements.txt   # 依赖清单
└── 学习说明.md         # 详细学习文档
```

## 快速开始

### 1. 克隆项目

```bash
git clone <你的仓库地址>
cd <项目目录>
```

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Windows 下激活虚拟环境：`.venv\Scripts\activate`

### 3. 配置 API Key

把 `.env.example` 复制为 `.env`，填入你的 DeepSeek API Key：

```bash
copy .env.example .env
```

```
DEEPSEEK_API_KEY=sk-你的Key
```

### 4. 启动服务

```bash
.venv\Scripts\python.exe -m uvicorn main:app --reload
```

浏览器访问 http://127.0.0.1:8000

### 5. 试试

- 100 美元等于多少人民币？
- 北京现在天气怎么样？
- 给我来一条名言

## 学习资源

想理解 Tool Calling 的原理和完整流程，请看 [学习说明.md](学习说明.md)。
