"""LangChain 聊天模型工厂。"""

import os
from typing import Any

from langchain.chat_models import init_chat_model

from config import ModelProfile


def build_chat_model(profile: ModelProfile) -> Any:
    """使用统一的 init_chat_model 创建 OpenAI 兼容聊天模型。"""
    api_key = os.getenv(profile.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"未找到 {profile.api_key_env}，请在 .env 中配置当前模型厂商的 API Key。"
        )

    return init_chat_model(
        profile.model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=profile.base_url,
        # 保留人设约束的同时给回复留出变化空间，避免同一模板反复出现。
        temperature=0.72,
        max_tokens=220,
    )
