"""心屿 LangGraph 五节点工作流。"""

from langgraph.graph import END, START, StateGraph

from nodes import (
    emotion_analysis_node,
    bond_analysis_node,
    memory_retrieval_node,
    persistence_node,
    response_generation_node,
)
from state import ConversationState


def build_workflow():
    """构建并编译情感分析、记忆检索、回复生成工作流。"""
    graph = StateGraph(ConversationState)
    graph.add_node("emotion_analysis", emotion_analysis_node)
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("bond_analysis", bond_analysis_node)
    graph.add_node("response_generation", response_generation_node)
    graph.add_node("persistence", persistence_node)

    # 两条分支共享 START，LangGraph 会在生成节点前等待两者完成。
    graph.add_edge(START, "emotion_analysis")
    graph.add_edge(START, "memory_retrieval")
    graph.add_edge("memory_retrieval", "response_generation")
    graph.add_edge("emotion_analysis", "bond_analysis")
    graph.add_edge("bond_analysis", "response_generation")
    graph.add_edge("response_generation", "persistence")
    graph.add_edge("persistence", END)
    return graph.compile()
