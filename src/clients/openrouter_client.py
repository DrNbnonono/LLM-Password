import os
import time
from typing import List, Optional, Dict, Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.verify_ssl = (os.getenv("OPENROUTER_VERIFY_SSL", "true").lower() != "false")
        # Disable reading proxy env by default to avoid system proxy interception
        self.trust_env = (os.getenv("OPENROUTER_TRUST_ENV", "false").lower() == "true")
        if not self.verify_ssl:
            try:
                import urllib3  # type: ignore
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

    @retry(stop=stop_after_attempt(7), wait=wait_exponential_jitter(initial=1, max=20))
    def chat_completions(self, model: str, messages: List[Dict[str, str]], n: int = 1, temperature: float = 0.7,
                         max_tokens: int = 64, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        x_title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if x_title:
            headers["X-Title"] = x_title
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        session = requests.Session()
        session.trust_env = self.trust_env
        response = session.post(
            url,
            json=payload,
            headers=headers,
            timeout=120,
            verify=self.verify_ssl,
            proxies={} if not self.trust_env else None,
        )
        if response.status_code >= 400:
            # Handle rate limiting with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                        time.sleep(min(max(sleep_s, 1.0), 30.0))
                    except Exception:
                        time.sleep(3.0)
                else:
                    time.sleep(3.0)
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
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
