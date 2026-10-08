"""SQLite 结构化数据与 Chroma 向量记忆的统一门面。"""

from pathlib import Path
import logging

from database import SQLiteStore, UserProfile
from vector_memory import ChromaLongTermMemory


logger = logging.getLogger("xinyu.memory")


class MemoryStore:
    """同时管理用户档案/历史和长期记忆向量。"""

    def __init__(
        self,
        database_path: str | Path = "data/xinyu.sqlite3",
        chroma_path: str | Path = "data/xinyu_chroma",
    ) -> None:
        self.database = SQLiteStore(database_path)
        self.vector = ChromaLongTermMemory(chroma_path)

    def add(self, content: str, metadata: dict[str, str] | None = None) -> None:
        self.vector.add(content, metadata)

    def search(self, query: str, user_id: str = "default_user", limit: int = 5) -> list[str]:
        """混合检索：结构化长期事实优先，再补充 Chroma 和历史。"""
        results: list[str] = []
        # 先查 memory_items，确保“我喜欢吃什么”不会被历史提问/客服追问遮住。
        query_lower = query.lower()
        for item in self.database.list_memory_items(user_id, limit=200):
            content = str(item.get("content", ""))
            if any(token in query_lower for token in ("喜欢", "爱吃", "偏好", "记得", "记住")):
                if "吃" in query_lower and "吃" not in content:
                    continue
                if "喝" in query_lower and "喝" not in content:
                    continue
                if any(token in content.lower() for token in ("喜欢", "爱吃", "偏好")):
                    results.append(content)
            elif any(term in content.lower() for term in query_lower.split() if term):
                results.append(content)
            if len(results) >= limit:
                break
        for item in self.database.search_history(query, user_id, limit=limit):
            if item not in results:
                results.append(item)
            if len(results) >= limit:
                break
        try:
            vector_results = self.vector.search(query, limit=limit, user_id=user_id)
        except Exception as exc:
            # SQLite 历史仍可用，向量库异常不应让整轮聊天失败。
            logger.warning("Chroma search unavailable; using SQLite memory only: %s", exc)
            vector_results = []
        for item in vector_results:
            if item not in results:
                results.append(item)
            if len(results) >= limit:
                break
        return results[:limit]

    def get_profile(self, user_id: str = "default_user") -> UserProfile:
        return self.database.get_profile(user_id)

    def load_history(self, user_id: str = "default_user", limit: int = 40):
        return self.database.load_history(user_id, limit)

    def list_memories(self, user_id: str = "default_user", limit: int = 50):
        return self.database.list_memory_items(user_id, limit)

    def delete_memory(self, user_id: str, memory_id: int) -> bool:
        return self.database.delete_memory_item(user_id, memory_id)

    def update_memory(self, user_id: str, memory_id: int, content: str) -> bool:
        return self.database.update_memory_item(user_id, memory_id, content)
