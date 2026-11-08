import requests, os, json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NIM_API_KEY")
model = os.getenv("NIM_MODEL")
print("Using model:", model)

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
}

try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("Status code:", r.status_code)
    print("Response:", r.text[:1000])
except Exception as e:
    print("Request failed:", type(e).__name__, e)