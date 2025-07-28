import json
import re

def extract_reverse_hyde_keys(text: str) -> list[str]:
    print('Running [extraction]')
    print('[response]:')
    print(text)
    match = re.search(r"<structured_response>(.*?)</structured_response>", text, re.DOTALL | re.IGNORECASE)
    return json.loads(match.group(1))