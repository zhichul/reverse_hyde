from openai import OpenAI

class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = OpenAI(api_key=self.cfg["api_key"])


    def complete(self, prompt: str) -> str:
        if 'o4' or 'o3' in self.cfg["model"]:
            if self.cfg["temperature"] != 0:
                print("[warning] o4 and o3 models can only be run with temperature 0, overriding")
            resp = self.client.chat.completions.create(
                model=self.cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                response_format=self.cfg["response_format"],
                max_completion_tokens=self.cfg["max_completion_tokens"],
            )
        else:
            resp = self.client.chat.completions.create(
                model=self.cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg["temperature"],
                response_format=self.cfg["response_format"],
                max_completion_tokens=self.cfg["max_completion_tokens"],
            )
        return resp.choices[0].message.content