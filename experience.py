"""次元模式的可配置角色皮肤与轻量剧情配置。"""

CHARACTER_SKINS: dict[str, dict[str, str]] = {
    "island": {
        "name": "小屿·岛灯",
        "description": "温柔、清晰、偶尔俏皮",
        "style": "保持温柔边界，使用简短而具体的陪伴表达。",
    },
    "stargazer": {
        "name": "小屿·星航",
        "description": "安静、想象力丰富、带一点宇宙感",
        "style": "可以使用轻量星空意象，但不要让意象盖过用户真实问题。",
    },
    "adventure": {
        "name": "小屿·旅伴",
        "description": "明快、行动导向、鼓励探索",
        "style": "用轻快语气提出一个可执行的小行动，不催促、不命令。",
    },
}


def list_skins() -> list[dict[str, str]]:
    return [{"id": skin_id, **skin} for skin_id, skin in CHARACTER_SKINS.items()]


def get_skin(skin_id: str) -> dict[str, str]:
    return CHARACTER_SKINS.get(skin_id, CHARACTER_SKINS["island"])


def story_snapshot(progress: int) -> dict[str, object]:
    progress = max(0, min(100, int(progress)))
    chapter = min(5, progress // 20 + 1)
    return {
        "progress": progress,
        "chapter": chapter,
        "title": f"岛屿日志 · 第 {chapter} 章",
        "next_hint": "继续记录一次真实近况" if progress < 100 else "你已经走过完整的岛屿旅程",
    }
