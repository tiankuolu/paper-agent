"""CLI for Paper Reading Agent.
论文阅读助手的命令行入口，启动后进入交互式对话循环。"""

import uuid  # 生成唯一会话 ID，用于 LangGraph 的记忆隔离
from dotenv import load_dotenv  # 从 .env 文件加载环境变量（API Key 等）
from rich.console import Console  # Rich 库的控制台，支持 Markdown 渲染和彩色输出
from rich.markdown import Markdown  # 把 AI 返回的 Markdown 文本渲染为终端富文本
from langchain_core.messages import HumanMessage  # LangChain 的人类消息类型，包装用户输入

load_dotenv()  # 把 .env 里的 LLM_PROVIDER、LLM_API_KEY 等注入到 os.environ
from src.agent import get_agent  # noqa: E402 - 读取配置后再加载 Agent

console = Console()  # 创建 Rich 控制台实例，后续所有输出都通过它
agent = get_agent()  # 使用 .env 中的活动模型配置构建 Agent
THREAD_ID = str(uuid.uuid4())  # 当前会话的唯一 thread_id，MemorySaver 用它区分不同对话

def main():
    """交互式主循环：读取用户输入 → agent 推理 → 渲染 Markdown 输出。"""
    # ---- 启动横幅 ----
    console.print("\n[bold cyan]📚 Paper Reading Agent[/bold cyan]")  # 青色粗体标题
    console.print("[dim]An AI-powered academic paper assistant[/dim]")  # 灰色副标题
    console.print("\nCommands: search → download → summarize / deep_read / ask")
    console.print("Type [red]quit[/red] to exit, [yellow]new[/yellow] for fresh session\n")

    while True:  # 无限循环，直到用户输入 quit 或 Ctrl+C
        try:
            user_input = console.input("[bold green]You > [/bold green]")  # 绿色提示符，等待用户输入
        except (EOFError, KeyboardInterrupt):  # 用户按 Ctrl+C 或 Ctrl+D
            break  # 退出循环

        if user_input.lower() in ("quit", "exit", "q"):  # 退出命令
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "new":  # 新建会话：换一个 thread_id，之前的记忆清空
            global THREAD_ID  # 声明要修改模块级变量
            THREAD_ID = str(uuid.uuid4())  # 生成全新的会话 ID
            console.print("[dim]✨ New session started[/dim]\n")
            continue  # 跳过本次循环，回到输入提示

        if not user_input.strip():  # 空输入（只按了回车），直接跳过
            continue

        console.print()  # 空一行，让输出和输入之间留白
        with console.status("[dim]Thinking...[/dim]"):  # 显示旋转动画 "Thinking..."，直到 agent 返回
            try:
                result = agent.invoke(  # 调用 LangGraph agent，传入当前消息列表
                    {"messages": [HumanMessage(content=user_input)]},  # 把用户输入包装成 HumanMessage
                    config={"configurable": {"thread_id": THREAD_ID}},  # 传入 thread_id 用于记忆
                )
                last_msg = result["messages"][-1]  # 取最后一条消息（AI 的回复）
                console.print(Markdown(last_msg.content))  # 用 Rich 渲染 Markdown 并打印
                console.print()  # 结尾空行
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")  # 红色打印异常信息

if __name__ == "__main__":  # 只在直接运行 python main.py 时执行，被 import 时不执行
    main()
