"""配置模块：负责加载环境变量、初始化 LLM、把工具绑定到 LLM 上。

这个文件是整个项目的「装配车间」：
  1. 从 .env 读取 API Key（保证密钥不写死在代码里）
  2. 创建 LLM 实例（指向 DeepSeek）
  3. 用 bind_tools 把工具挂到 LLM 上
  4. 生成一个「工具名 -> 工具对象」的映射，供 main.py 执行工具时查找
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

# 把 .env 文件里的内容加载成环境变量。
# 这样 os.getenv("DEEPSEEK_API_KEY") 才能读到 Key。
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 如果没配 Key，在启动时立刻报错并给出提示，
# 而不是等到用户发消息时才莫名其妙地失败。
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY。请先复制 .env.example 为 .env，"
        "并填入你的 DeepSeek API Key。"
    )

# 初始化 LLM。
# DeepSeek 提供了「OpenAI 兼容」的接口，所以能直接复用 langchain-openai 的
# ChatOpenAI，只要把 base_url 指到 DeepSeek 的地址即可，不需要单独装 DeepSeek 的包。
llm = ChatOpenAI(
    model="deepseek-chat",                # DeepSeek 的对话模型名
    api_key=DEEPSEEK_API_KEY,             # 从环境变量读取的 Key
    base_url="https://api.deepseek.com",  # DeepSeek 的 OpenAI 兼容地址
    temperature=0.7,                      # 随机性：0 最稳定，越大越发散
)

# 关键一步：bind_tools 把工具「绑定」到 LLM 上。
#
# 它做了什么？之后每次调用 llm_with_tools.invoke(...) 时，LangChain 会自动
# 把我们定义的每个工具的名字、描述、参数说明打包成一份 JSON（工具说明书），
# 随用户的问题一起发给 LLM。
#
# LLM 看到这份说明书后，就能自主判断：
#   - 需不需要用工具？
#   - 用哪个工具？
#   - 给这个工具传什么参数？
#
# 这就是 Tool Calling 的核心机制。
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# 建立「工具名 -> 工具对象」的映射。
# LLM 返回的是一串字符串工具名（比如 "get_weather"），
# 我们靠这个映射把它转换成真正的 Python 对象，才能执行它。
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
