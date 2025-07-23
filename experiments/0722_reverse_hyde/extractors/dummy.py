import json


def extract_reverse_hyde_keys(text: str) -> list[str]:
    return json.loads(text)['interesting_ideas']