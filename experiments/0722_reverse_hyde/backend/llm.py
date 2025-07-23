from openai import OpenAI

class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = OpenAI(api_key=self.cfg["api_key"])


    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg["temperature"],
            response_format=self.cfg["response_format"],
            max_completion_tokens=self.cfg["max_completion_tokens"],
        )
        return resp.choices[0].message.content