import os
from typing import List, Optional, Dict, Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class SiliconFlowClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.base_url = (base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1").rstrip("/")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is not set.")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=10))
    def chat_completions(self, model: str, messages: List[Dict[str, str]], n: int = 1, temperature: float = 0.7,
                         max_tokens: int = 64) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"SiliconFlow error {response.status_code}: {response.text}")
        return response.json()

    @staticmethod
    def _system_prompt(lang: str = "both") -> str:
        zh = "你是一个口令生成器。只输出口令字符串。"
        en = "You are a password generator. Output only the password string."
        lang = (lang or "both").lower()
        if lang == "zh":
            return zh
        if lang == "en":
            return en
        return zh + " \n" + en

    def generate_passwords(self, model: str, prompt: str, count: int = 1, temperature: float = 0.6,
                            max_tokens: int = 32, lang: str = "both") -> List[str]:
        messages = [
            {"role": "system", "content": self._system_prompt(lang)},
            {"role": "user", "content": prompt},
        ]
        data = self.chat_completions(model=model, messages=messages, n=count, temperature=temperature,
                                     max_tokens=max_tokens)
        passwords: List[str] = []
        for choice in data.get("choices", []):
            content = choice.get("message", {}).get("content", "").strip()
            first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
            if first_line:
                passwords.append(first_line)
        return passwords
