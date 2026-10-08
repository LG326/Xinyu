"""基于 LangGraph 的心屿终端对话程序。"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import get_model_profile
from memory_store import MemoryStore
from model_factory import build_chat_model
from state import ConversationState
from workflow import build_workflow


def chat_loop() -> None:
    load_dotenv()
    provider, model_profile = get_model_profile()
    chat_model = build_chat_model(model_profile)
    memory_store = MemoryStore()
    workflow = build_workflow()
    user_id = "default_user"
    user_profile = memory_store.get_profile(user_id)
    history: list[BaseMessage] = memory_store.load_history(user_id)
    bond_value = user_profile.bond_value

    print(f"当前模型：{provider} / {model_profile.model_name}")
    print(f"当前羁绊值：{bond_value:g}/100")
    print("小屿：我在。今天想从哪件事说起？")
    print("（输入 exit 退出）")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n小屿：知道了。先去忙吧。")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("小屿：知道了。先去忙吧。")
            break

        state: ConversationState = {
            "user_input": user_input,
            "history": history,
            "bond_value": bond_value,
            "mode": "reality",
            "companion_alias": "小屿",
        }
        try:
            result = workflow.invoke(
                state,
                config={
                    "configurable": {
                        "chat_model": chat_model,
                        "memory_store": memory_store,
                        "user_id": user_id,
                    }
                },
            )
            assistant_text = result["response"]
        except Exception as exc:
            print(f"小屿暂时没接上话：{exc}")
            continue

        history.extend(
            [
                HumanMessage(content=user_input),
                AIMessage(content=assistant_text),
            ]
        )
        bond_value = float(result.get("bond_value", bond_value))
        if len(history) > 40:
            history = history[-40:]
        print(f"小屿：{assistant_text}")
        print(f"（羁绊值 {bond_value:g}/100，本轮变化 {float(result.get('bond_delta', 0.0)):+g}）")


if __name__ == "__main__":
    try:
        chat_loop()
    except RuntimeError as exc:
        print(f"配置错误：{exc}")
