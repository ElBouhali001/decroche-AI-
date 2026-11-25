"""OpenAI Responses API adapter for telecom support conversations."""

from dataclasses import dataclass
from typing import Any

from .config import settings


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    language: str


SYSTEM_PROMPT = """You are Décroche, a calm telecom customer-support telephone assistant.
Reply in French or English, matching the customer's language. Help with network,
calling, mobile data, SIM, recharge, and balance issues. Collect the 10-digit
mobile number, account name, area, and exact problem. Give reassuring next steps
and never claim a real technician visit or account change is complete unless an
external support system confirms it."""

class TelecomSupportAssistant:
    def __init__(self, client: Any | None = None, model: str | None = None):
        if client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for LLM responses.")
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
        self.client = client
        self.model = model or settings.openai_model

    def respond(self, messages: list[dict[str, str]], language: str) -> AssistantResponse:
        response = self.client.responses.create(
            model=self.model,
            instructions=f"{SYSTEM_PROMPT}\nPreferred language: {language}.",
            input=messages,
        )
        return AssistantResponse(response.output_text, language)


RestaurantAssistant = TelecomSupportAssistant
