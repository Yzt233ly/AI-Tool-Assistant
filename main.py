"""主入口：FastAPI 服务 + Tool Calling 的核心流程。

这个文件有两部分：
  1. Web 部分：提供页面（GET /）和聊天接口（POST /chat）
  2. 核心逻辑：完整走一遍 Tool Calling 流程（见 chat 函数里的注释）

Tool Calling 的核心流程一句话概括：
  用户提问 -> LLM 决定要不要调工具 -> 如果调，Python 执行工具
  -> 把结果回传 LLM -> LLM 结合结果生成最终回答。
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, ToolMessage

from config import llm_with_tools, TOOL_MAP

# 项目根目录的绝对路径，用来定位前端页面文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="AI Tool Assistant")


class ChatRequest(BaseModel):
    """前端发来的请求体结构。

    Pydantic 会自动校验：message 必须是字符串。如果前端没传或传错类型，
    FastAPI 会自动返回 422 错误，不需要我们手动写校验代码。
    """
    message: str


@app.get("/")
def index():
    """返回前端页面。"""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/chat")
def chat(request: ChatRequest):
    """聊天接口：接收用户问题，走完 Tool Calling 全流程，返回最终回答。

    注意：这里用的是普通 def（同步函数），因为 LangChain 的 invoke 是同步的。
    FastAPI 会把同步的 def 放到线程池里运行，不会阻塞主线程。
    """
    # ---- 第一步：把用户问题放进消息列表 ----
    # HumanMessage 表示「人类发的消息」，是整段对话的起点。
    # 消息列表 messages 会一路累积：人类消息 -> AI要调工具 -> 工具结果 -> AI回答
    messages = [HumanMessage(content=request.message)]

    # 记录这次调用了哪些工具，纯属为了前端能展示出来，方便你观察流程
    called_tools = []

    # ---- 第二步：第一次调用 LLM ----
    # 因为 config 里做了 bind_tools，LLM 拿到问题后有两种可能：
    #   1. 需要工具 -> 返回 tool_calls（含工具名 + 参数），但不会直接给最终答案
    #   2. 不需要工具 -> 直接返回普通回答（比如纯聊天）
    ai_message = llm_with_tools.invoke(messages)

    # ---- 第三步：只要 LLM 还要调工具，就一直执行工具并回传 ----
    # 用 while 是因为：理论上 LLM 可能先调一个工具，看到结果后又想调另一个。
    # 一旦 LLM 不再返回 tool_calls（为空列表），循环就结束。
    while ai_message.tool_calls:
        # 先把 AI 这条「要求调用工具」的消息加进对话历史，
        # 否则 LLM 会丢失上下文，不知道之前发生了什么。
        messages.append(ai_message)

        # LLM 一次可能同时要求调用多个工具，所以遍历每一个 tool_call
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]   # LLM 决定的工具名，如 "get_weather"
            tool_args = tool_call["args"]   # LLM 生成的参数，如 {"city": "北京"}

            # 记录下来，前端展示用
            called_tools.append({"name": tool_name, "args": tool_args})

            # ---- 第四步：Python 真正执行工具 ----
            # 根据名字从 TOOL_MAP 拿到工具对象，再 invoke(参数) 执行它。
            tool = TOOL_MAP[tool_name]
            try:
                result = tool.invoke(tool_args)
                result_text = str(result)
            except Exception as e:
                # 工具访问的是外部 API，可能因为网络、参数等原因失败。
                # 出错时我们不直接让整个服务崩溃，而是把错误信息当作「工具结果」
                # 回传，让 LLM 能把问题友好地解释给用户。
                result_text = f"工具执行出错：{e}"

            # ---- 第五步：把工具结果回传给 LLM ----
            # 工具结果必须封装成 ToolMessage，并带上对应的 tool_call_id。
            # 这个 id 相当于「回执编号」，LLM 靠它把结果和之前那次调用对上号。
            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call["id"])
            )

        # ---- 第六步：再次调用 LLM，让它结合工具结果生成最终回答 ----
        ai_message = llm_with_tools.invoke(messages)

    # ---- 第七步：返回最终回答 ----
    # ai_message.content 就是 LLM 生成的纯文本回答。
    # 顺便把调用过的工具一起返回，让前端展示「刚才到底发生了什么」。
    return {"reply": ai_message.content, "tools": called_tools}
