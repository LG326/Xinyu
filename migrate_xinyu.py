"""将旧心屿 SQLite 数据库迁移到新命名。

用法：python migrate_xinyu.py [旧数据库] [新数据库]
脚本只复制并迁移到新文件，默认保留旧文件不变。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from database import SQLiteStore


def migrate(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"找不到旧数据库：{source}")
    if destination.exists():
        raise FileExistsError(f"目标已存在，为避免覆盖请先备份或更换路径：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    SQLiteStore(destination)
    print(f"迁移完成：{source} -> {destination}")
    print("旧文件未修改；请在校验通过后再切换 SQLITE_PATH。")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/liu_ruyan.sqlite3")
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/xinyu.sqlite3")
    migrate(source, destination)
