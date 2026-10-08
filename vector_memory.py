"""Chroma 长期记忆向量存储。"""

import hashlib
import logging
import math
import re
from pathlib import Path

from langchain_core.embeddings import Embeddings


logger = logging.getLogger("xinyu.memory")


class HashEmbeddings(Embeddings):
    """无需额外模型服务的确定性本地嵌入，用于阶段四验证向量检索链路。"""

    dimensions = 256

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        chars = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())
        terms = chars + ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]
        vector = [0.0] * cls.dimensions
        for term in terms:
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % cls.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ChromaLongTermMemory:
    """将长期记忆写入持久化 Chroma collection，并进行相似度检索。"""

    def __init__(self, path: str | Path = "data/xinyu_chroma") -> None:
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Chroma 依赖，请执行：pip install -r requirements.txt"
            ) from exc

        self.store = Chroma(
            collection_name="xinyu_long_term_memory",
            persist_directory=str(path),
            embedding_function=HashEmbeddings(),
        )

    def add(self, content: str, metadata: dict[str, str] | None = None) -> None:
        if content.strip():
            self.store.add_texts([content.strip()], metadatas=[metadata or {}])

    def search(
        self, query: str, limit: int = 3, user_id: str | None = "default_user"
    ) -> list[str]:
        # langchain-chroma 不同小版本对过滤参数支持不一致。过滤失败时不能退化为
        # 全量检索，否则会把其他用户的长期记忆混入当前对话；上层 SQLite 仍会提供
        # 当前用户的结构化历史作为安全兜底。
        try:
            if user_id:
                documents = self.store.similarity_search(
                    query, k=limit, filter={"user_id": user_id}
                )
            else:
                documents = self.store.similarity_search(query, k=limit)
        except Exception as exc:
            logger.warning("Chroma filtered search unavailable; skipping vector memory: %s", exc)
            return []
        # 仅使用新格式的用户事实记忆。旧版本把用户输入和助手回复拼在一起，
        # 其中常含“你喜欢哪种？”等追问；忽略这类无 role 标记的旧向量，
        # 避免它们继续污染事实召回，同时不删除已有索引。
        filtered = [
            document
            for document in documents
            if (getattr(document, "metadata", {}) or {}).get("role") == "user_fact"
            and (not user_id or (getattr(document, "metadata", {}) or {}).get("user_id") == user_id)
        ]
        return [document.page_content for document in filtered]
