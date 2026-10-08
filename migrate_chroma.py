"""迁移 Chroma 长期记忆 collection，默认只复制用户事实记忆。

用法：python migrate_chroma.py [持久化目录]
源 collection 保留不动，目标为 ``xinyu_long_term_memory``。
"""

from __future__ import annotations

import sys
from pathlib import Path

from vector_memory import HashEmbeddings


def migrate(path: Path) -> int:
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("缺少 Chroma 依赖，请先安装 requirements.txt") from exc

    source = Chroma(
        collection_name="liu_ruyan_long_term_memory",
        persist_directory=str(path),
        embedding_function=HashEmbeddings(),
    )
    target = Chroma(
        collection_name="xinyu_long_term_memory",
        persist_directory=str(path),
        embedding_function=HashEmbeddings(),
    )
    payload = source.get(include=["documents", "metadatas"])
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    selected: list[tuple[str, dict[str, str]]] = []
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        if metadata.get("role") != "user_fact":
            continue
        selected.append((document, dict(metadata)))
    if selected:
        target.add_texts(
            [document for document, _ in selected],
            metadatas=[metadata for _, metadata in selected],
        )
    print(f"迁移完成：{len(selected)} 条用户事实记忆 -> xinyu_long_term_memory")
    print("旧 collection 未删除；请抽样验证召回后再清理。")
    return len(selected)


if __name__ == "__main__":
    migrate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/xinyu_chroma"))
