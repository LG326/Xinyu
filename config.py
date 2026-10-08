"""模型厂商配置与选择逻辑。"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """一个 OpenAI 兼容模型端点的连接配置。"""

    base_url: str
    model_name: str
    api_key_env: str


# 所有端点均通过 langchain-openai 调用，新增厂商时只需在这里增加配置。
MODEL_PROFILES: dict[str, ModelProfile] = {
    "deepseek": ModelProfile(
        base_url="https://api.deepseek.com",
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen": ModelProfile(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "zhipu": ModelProfile(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_name="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
    ),
}


def get_model_profile() -> tuple[str, ModelProfile]:
    """根据 MODEL_PROVIDER 返回厂商名称和对应配置。"""
    provider = os.getenv("MODEL_PROVIDER", "deepseek").strip().lower()
    try:
        return provider, MODEL_PROFILES[provider]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_PROFILES))
        raise RuntimeError(
            f"不支持的 MODEL_PROVIDER={provider!r}，可选值：{supported}。"
        ) from exc
