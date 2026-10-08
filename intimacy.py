"""羁绊值信号识别与陪伴语气策略。

文件名暂时保留以兼容已有部署；新代码使用 Bond* 命名。
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BondSignal:
    requested_delta: float
    event: str
    reconnection_hint: bool = False


DAILY_WORDS = ("今天", "昨天", "早上", "午饭", "晚饭", "上班", "下班", "周末", "奶茶", "猫")
CARE_WORDS = ("关心", "想你", "辛苦", "吃饭", "早点休息", "照顾好", "陪你", "谢谢你")
DEEP_WORDS = ("需要你", "陪陪我", "撑不住", "很难受", "一直睡不着", "没人理解")
RECONNECTION_WORDS = ("对不起", "抱歉", "和好", "原谅我", "别生气", "我错了")


def calculate_bond_signal(
    text: str, emotion: str, last_interaction_at: datetime | None
) -> BondSignal:
    """按规则计算本轮羁绊反馈；实际落分由 SQLite 的每日上限控制。"""
    normalized = text.lower()
    if any(word in normalized for word in RECONNECTION_WORDS):
        return BondSignal(1.0, "reconnection")

    delta = 0.0
    events: list[str] = []
    if any(word in normalized for word in DAILY_WORDS):
        delta += 1.0
        events.append("daily_sharing")
    if any(word in normalized for word in CARE_WORDS):
        delta += 1.0
        events.append("care_expression")
    if emotion in {"难过", "焦虑", "生气"} or any(
        word in normalized for word in DEEP_WORDS
    ):
        delta += 2.0 if any(word in normalized for word in DEEP_WORDS) else 1.0
        events.append("emotional_support")

    if last_interaction_at and (datetime.now() - last_interaction_at).total_seconds() > 24 * 3600:
        return BondSignal(delta, "+".join(events) or "reconnection_hint", reconnection_hint=True)
    return BondSignal(delta, "+".join(events) or "neutral")


def bond_tone(score: float, reconnection_hint: bool = False) -> str:
    if reconnection_hint:
        return "温和接住近况，带一点重新连接的提示；不责备、不索取"
    if score >= 70:
        return "温柔核心不变，可以多接一句并保持轻松俏皮，不制造依赖"
    if score >= 50:
        return "保持边界，先回应再自然接一个相关话题，不主动讨好"
    return "清晰、温和、简短地接住话题，必要时给出实际的小帮助"


# 兼容旧模块调用；新代码应使用 Bond* 命名。
IntimacySignal = BondSignal
calculate_signal = calculate_bond_signal
intimacy_tone = bond_tone
