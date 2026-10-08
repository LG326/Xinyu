"""SQLite 持久化：用户档案、对话历史和羁绊事件。"""

import sqlite3
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    display_name: str
    bond_value: float
    daily_bond_gain: float
    bond_gain_date: str
    last_interaction_at: str | None
    mode: str = "reality"
    companion_alias: str = "小屿"
    skin_id: str = "island"
    story_progress: int = 0
    reminder_frequency: str = "standard"
    total_active_seconds: int = 0

    # 旧属性只读兼容，避免旧客户端升级时崩溃。
    @property
    def intimacy(self) -> float:
        return self.bond_value

    @property
    def daily_positive(self) -> float:
        return self.daily_bond_gain

    @property
    def gain_date(self) -> str:
        return self.bond_gain_date

    @property
    def last_user_at(self) -> str | None:
        return self.last_interaction_at


class SQLiteStore:
    """管理心屿应用的结构化持久化数据。"""

    def __init__(self, path: str | Path = "data/xinyu.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = OFF")
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(user_profiles)").fetchall()
            }
            if "intimacy" in columns:
                self._migrate_legacy_schema(connection)
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(user_profiles)").fetchall()
                }
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    bond_value REAL NOT NULL DEFAULT 40,
                    daily_bond_gain REAL NOT NULL DEFAULT 0,
                    bond_gain_date TEXT NOT NULL,
                    last_interaction_at TEXT,
                    mode TEXT NOT NULL DEFAULT 'reality' CHECK (mode IN ('reality', 'dimensional')),
                    companion_alias TEXT NOT NULL DEFAULT '小屿',
                    skin_id TEXT NOT NULL DEFAULT 'island',
                    story_progress INTEGER NOT NULL DEFAULT 0,
                    reminder_frequency TEXT NOT NULL DEFAULT 'standard',
                    total_active_seconds INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    emotion TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                CREATE TABLE IF NOT EXISTS bond_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    requested_delta REAL NOT NULL,
                    applied_delta REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                CREATE TABLE IF NOT EXISTS interaction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    confidence REAL NOT NULL DEFAULT 0.8,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_history_user_created
                    ON conversation_history(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_bond_events_user_created
                    ON bond_events(user_id, created_at);
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(user_profiles)").fetchall()
            }
            for column, definition in {
                "mode": "TEXT NOT NULL DEFAULT 'reality'",
                "companion_alias": "TEXT NOT NULL DEFAULT '小屿'",
                "skin_id": "TEXT NOT NULL DEFAULT 'island'",
                "story_progress": "INTEGER NOT NULL DEFAULT 0",
                "reminder_frequency": "TEXT NOT NULL DEFAULT 'standard'",
                "total_active_seconds": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE user_profiles ADD COLUMN {column} {definition}")
            memory_columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_items)").fetchall()}
            for column, definition in {"memory_type": "TEXT NOT NULL DEFAULT 'fact'", "confidence": "REAL NOT NULL DEFAULT 0.8"}.items():
                if column not in memory_columns:
                    connection.execute(f"ALTER TABLE memory_items ADD COLUMN {column} {definition}")
            connection.execute("PRAGMA user_version = 1")
            connection.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
        """将旧柳如烟 schema 原地迁移为心屿 schema；不删除用户数据。"""
        connection.execute("ALTER TABLE user_profiles RENAME TO user_profiles_legacy")
        connection.execute("ALTER TABLE conversation_history RENAME TO conversation_history_legacy")
        connection.execute("ALTER TABLE intimacy_events RENAME TO bond_events_legacy")
        connection.execute(
            """
            CREATE TABLE user_profiles (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                bond_value REAL NOT NULL DEFAULT 40,
                daily_bond_gain REAL NOT NULL DEFAULT 0,
                bond_gain_date TEXT NOT NULL,
                last_interaction_at TEXT,
                mode TEXT NOT NULL DEFAULT 'reality',
                companion_alias TEXT NOT NULL DEFAULT '小屿',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        legacy_user = "司序"
        default_exists = connection.execute(
            "SELECT 1 FROM user_profiles_legacy WHERE user_id = 'default_user'"
        ).fetchone() is not None
        connection.execute(
            """
            INSERT INTO user_profiles
            (user_id, display_name, bond_value, daily_bond_gain, bond_gain_date,
             last_interaction_at, mode, companion_alias, created_at, updated_at)
            SELECT CASE WHEN user_id = ? THEN 'default_user' ELSE user_id END,
                   CASE WHEN user_id = ? THEN '你' ELSE display_name END,
                   intimacy, daily_positive, gain_date, last_user_at,
                   'reality', '小屿', created_at, updated_at
            FROM user_profiles_legacy
            WHERE NOT (user_id = ? AND ? IS NOT NULL)
            """,
            (legacy_user, legacy_user, legacy_user, 1 if default_exists else None),
        )
        if default_exists:
            # 已存在 default_user 时，旧“司序”档案并入该档案，避免主键冲突；
            # 对话和事件仍在下方完整迁移。
            connection.execute(
                """
                UPDATE user_profiles
                SET bond_value = MAX(
                        bond_value,
                        COALESCE((SELECT intimacy FROM user_profiles_legacy WHERE user_id = ?), bond_value)
                    ),
                    last_interaction_at = COALESCE(
                        (SELECT last_user_at FROM user_profiles_legacy WHERE user_id = ?),
                        last_interaction_at
                    )
                WHERE user_id = 'default_user'
                """,
                (legacy_user, legacy_user),
            )
        connection.execute(
            """
            CREATE TABLE conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                emotion TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO conversation_history (id, user_id, role, content, emotion, created_at)
            SELECT id, CASE WHEN user_id = '司序' THEN 'default_user' ELSE user_id END,
                   role, content, emotion, created_at
            FROM conversation_history_legacy
            """
        )
        connection.execute(
            """
            CREATE TABLE bond_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                requested_delta REAL NOT NULL,
                applied_delta REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO bond_events (id, user_id, event, requested_delta, applied_delta, created_at)
            SELECT id, CASE WHEN user_id = '司序' THEN 'default_user' ELSE user_id END,
                   event, requested_delta, applied_delta, created_at
            FROM bond_events_legacy
            """
        )
        connection.execute("DROP TABLE conversation_history_legacy")
        connection.execute("DROP TABLE bond_events_legacy")
        connection.execute("DROP TABLE user_profiles_legacy")
        connection.execute("PRAGMA user_version = 1")

    def get_profile(self, user_id: str = "default_user") -> UserProfile:
        today = datetime.now().date().isoformat()
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO user_profiles
                    (user_id, display_name, bond_value, daily_bond_gain, bond_gain_date,
                     created_at, updated_at)
                    VALUES (?, ?, 40, 0, ?, ?, ?)
                    """,
                    (user_id, "你" if user_id == "default_user" else user_id, today, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
                ).fetchone()
            elif row["bond_gain_date"] != today:
                connection.execute(
                    "UPDATE user_profiles SET daily_bond_gain = 0, bond_gain_date = ?, updated_at = ? WHERE user_id = ?",
                    (today, now, user_id),
                )
                row = connection.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
                ).fetchone()

        return UserProfile(
            user_id=row["user_id"],
            display_name=row["display_name"],
            bond_value=row["bond_value"],
            daily_bond_gain=row["daily_bond_gain"],
            bond_gain_date=row["bond_gain_date"],
            last_interaction_at=row["last_interaction_at"],
            mode=row["mode"],
            companion_alias=row["companion_alias"],
            skin_id=row["skin_id"],
            story_progress=row["story_progress"],
            reminder_frequency=row["reminder_frequency"],
            total_active_seconds=row["total_active_seconds"],
        )

    def get_last_interaction_at(self, user_id: str = "default_user") -> datetime | None:
        value = self.get_profile(user_id).last_interaction_at
        return datetime.fromisoformat(value) if value else None

    get_last_user_at = get_last_interaction_at

    def set_preferences(
        self,
        user_id: str,
        *,
        mode: str | None = None,
        companion_alias: str | None = None,
        skin_id: str | None = None,
        reminder_frequency: str | None = None,
    ) -> None:
        """保存用户选择的模式和小屿称呼。"""
        if mode not in (None, "reality", "dimensional"):
            raise ValueError("mode must be reality or dimensional")
        alias = companion_alias.strip() if companion_alias else None
        if alias is not None and not alias:
            alias = "小屿"
        assignments: list[str] = []
        params: list[str] = []
        if mode is not None:
            assignments.append("mode = ?")
            params.append(mode)
        if alias is not None:
            assignments.append("companion_alias = ?")
            params.append(alias[:32])
        if skin_id is not None:
            assignments.append("skin_id = ?")
            params.append(skin_id[:32])
        if reminder_frequency is not None:
            if reminder_frequency not in ("low", "standard", "high"):
                raise ValueError("reminder_frequency must be low, standard or high")
            assignments.append("reminder_frequency = ?")
            params.append(reminder_frequency)
        if not assignments:
            return
        now = datetime.now().isoformat(timespec="seconds")
        assignments.append("updated_at = ?")
        params.extend([now, user_id])
        with self._connect() as connection:
            connection.execute(
                f"UPDATE user_profiles SET {', '.join(assignments)} WHERE user_id = ?",
                params,
            )

    def get_usage_stats(self, user_id: str = "default_user") -> dict[str, int]:
        """返回用于反沉迷提示的轻量使用统计，不记录额外行为数据。"""
        with self._connect() as connection:
            today_count = connection.execute(
                "SELECT COUNT(*) FROM conversation_history "
                "WHERE user_id = ? AND role = 'user' AND date(created_at) = date('now', 'localtime')",
                (user_id,),
            ).fetchone()[0]
            hour_count = connection.execute(
                "SELECT COUNT(*) FROM conversation_history "
                "WHERE user_id = ? AND role = 'user' "
                "AND datetime(created_at) >= datetime('now', 'localtime', '-1 hour')",
                (user_id,),
            ).fetchone()[0]
        return {"today_messages": int(today_count), "last_hour_messages": int(hour_count)}

    def add_memory_item(self, user_id: str, content: str, emotion: str = "", source: str = "conversation", memory_type: str = "fact", confidence: float = 0.8) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_items (user_id, content, emotion, source, memory_type, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, content.strip(), emotion, source, memory_type, max(0.0, min(1.0, float(confidence))), now),
            )
            return int(cursor.lastrowid)

    def list_memory_items(self, user_id: str = "default_user", limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, content, emotion, source, memory_type, confidence, created_at FROM memory_items WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        # 兼容早期版本曾把每条聊天都写入 memory_items 的情况：面板只展示
        # 明确的事实陈述，避免“你好/我喜欢吃什么”等问题冒充长期记忆。
        fact_markers = ("我喜欢", "用户喜欢", "我爱吃", "我养了", "我有一只", "我最近在", "我上周")
        filtered = []
        for row in rows:
            content = str(row["content"])
            if any(marker in content for marker in fact_markers) and not any(mark in content for mark in ("什么", "吗", "？", "?")):
                filtered.append(dict(row))
            if len(filtered) >= limit:
                break
        return filtered

    def delete_memory_item(self, user_id: str, memory_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memory_items WHERE id = ? AND user_id = ?", (memory_id, user_id))
            return cursor.rowcount > 0

    def update_memory_item(self, user_id: str, memory_id: int, content: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE memory_items SET content = ? WHERE id = ? AND user_id = ?",
                (content.strip(), memory_id, user_id),
            )
            return cursor.rowcount > 0

    def record_interaction(self, user_id: str, event_type: str, payload: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO interaction_events (user_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (user_id, event_type, payload, now),
            )

    def add_active_seconds(self, user_id: str, seconds: int) -> int:
        seconds = max(0, min(int(seconds), 86400))
        with self._connect() as connection:
            connection.execute(
                "UPDATE user_profiles SET total_active_seconds = total_active_seconds + ?, updated_at = ? WHERE user_id = ?",
                (seconds, datetime.now().isoformat(timespec="seconds"), user_id),
            )
        return self.get_profile(user_id).total_active_seconds

    def list_interactions(self, user_id: str = "default_user", limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, event_type, payload, created_at FROM interaction_events WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def advance_story(self, user_id: str, mode: str) -> int:
        if mode != "dimensional":
            return self.get_profile(user_id).story_progress
        with self._connect() as connection:
            connection.execute(
                "UPDATE user_profiles SET story_progress = MIN(100, story_progress + 1), updated_at = ? WHERE user_id = ?",
                (datetime.now().isoformat(timespec="seconds"), user_id),
            )
        return self.get_profile(user_id).story_progress

    def apply_bond_delta(
        self,
        requested_delta: float,
        event: str,
        user_id: str = "default_user",
    ) -> tuple[float, float]:
        """应用本轮变化，正向变化每日最多累计 3 分。"""
        profile = self.get_profile(user_id)
        allowed_positive = max(0.0, 3.0 - profile.daily_bond_gain)
        positive = min(max(requested_delta, 0.0), allowed_positive)
        negative = min(requested_delta, 0.0)
        applied_delta = positive + negative
        bond_value = min(100.0, max(0.0, profile.bond_value + applied_delta))
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE user_profiles
                SET bond_value = ?, daily_bond_gain = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    bond_value,
                    profile.daily_bond_gain + max(applied_delta, 0.0),
                    now,
                    user_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO bond_events
                (user_id, event, requested_delta, applied_delta, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, event, requested_delta, applied_delta, now),
            )
        return bond_value, applied_delta

    apply_intimacy = apply_bond_delta

    def save_turn(
        self,
        user_input: str,
        assistant_text: str,
        emotion: str,
        user_id: str = "default_user",
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_history
                (user_id, role, content, emotion, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (user_id, "user", user_input, emotion, now),
                    (user_id, "assistant", assistant_text, emotion, now),
                ],
            )
            connection.execute(
                "UPDATE user_profiles SET last_interaction_at = ?, updated_at = ? WHERE user_id = ?",
                (now, now, user_id),
            )

    def load_history(
        self, user_id: str = "default_user", limit: int = 40
    ) -> list[BaseMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM conversation_history
                WHERE user_id = ? ORDER BY id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        messages: list[BaseMessage] = []
        for row in reversed(rows):
            if row["role"] == "user":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))
        return messages

    def search_history(
        self, query: str, user_id: str = "default_user", limit: int = 5
    ) -> list[str]:
        """按当前输入找相关记忆，并优先返回用户说过的事实。

        旧实现按 ``ORDER BY id DESC`` 直接截取结果，导致“我喜欢吃什么”
        这类问题会被过去相同的提问/追问占满，真正的偏好陈述反而出不来。
        这里先扩大候选集，再做轻量相关度排序：用户消息优先，疑问句降权，
        包含更多关键词且更具体的陈述优先。这样不依赖额外 embedding 服务，
        SQLite 也能稳定完成事实召回。
        """
        query = query.strip()
        if not query:
            return []
        raw_terms = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query.lower())
        terms: list[str] = []
        for raw in raw_terms:
            if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
                terms.extend(raw[index : index + 2] for index in range(len(raw) - 1))
            else:
                terms.append(raw)
        terms = list(dict.fromkeys(terms))[:6]
        clauses = ["content LIKE ?" for _ in terms]
        params: list[str | int] = [user_id, *[f"%{term}%" for term in terms]]
        if not clauses:
            clauses = ["content LIKE ?"]
            params.append(f"%{query}%")
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, role, content FROM conversation_history
                WHERE user_id = ? AND ({' OR '.join(clauses)})
                ORDER BY id DESC LIMIT ?
                """,
                (*params, max(limit * 12, 60)),
            ).fetchall()

        query_lower = query.lower()
        preference_query = any(
            marker in query_lower for marker in ("喜欢", "爱吃", "偏好", "习惯", "记得", "记住")
        )
        question_markers = ("什么", "吗", "呢", "哪", "怎么", "？", "?")
        scored: list[tuple[float, int, str]] = []
        seen_contents: set[str] = set()
        for row in rows:
            content = str(row["content"])
            content_lower = content.lower()
            normalized_content = re.sub(r"\s+", "", content_lower)
            if normalized_content in seen_contents:
                continue
            matched = sum(1 for term in terms if term and term in content_lower)
            if matched == 0:
                continue
            # “我喜欢吃什么/你还记得吗”是检索请求，不是用户事实；
            # 偏好类查询优先找陈述句，避免把同一问题的历史回声塞回上下文。
            is_question = any(marker in content_lower for marker in question_markers)
            if preference_query and is_question:
                continue
            # “我喜欢吃什么”要找吃的偏好；仅包含“喜欢你/喜欢电影”等泛喜欢句
            # 不应因为共享“喜欢”二字被召回。
            if preference_query and "吃" in query_lower and "吃" not in content_lower:
                continue
            if preference_query and row["role"] != "user":
                continue
            score = float(matched * 10)
            # 用户自己的陈述是长期记忆的主要来源；模型追问只作补充证据。
            if row["role"] == "user":
                score += 5
            else:
                score -= 2
            # 查询本身或“你喜欢什么？”等疑问句不是事实，避免循环召回。
            if is_question:
                score -= 7
            if content_lower == query_lower:
                score -= 8
            if preference_query and row["role"] == "user" and not any(
                marker in content_lower for marker in question_markers
            ):
                score += 12
            # 同等相关度时保留较新的记录。
            scored.append((score, int(row["id"]), f"{('用户' if row['role'] == 'user' else '小屿')}：{content}"))
            seen_contents.add(normalized_content)

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]
