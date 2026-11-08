import requests, os, json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NIM_API_KEY")

candidates = [
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "microsoft/phi-3.5-moe-instruct",
    "google/gemma-2b",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mixtral-8x22b-v0.1",
]

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

for model in candidates:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 20,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"{model} -> {r.status_code}")
        if r.status_code == 200:
            print("   WORKS:", r.json()["choices"][0]["message"]["content"][:80])
    except Exception as e:
        print(f"{model} -> ERROR: {e}")
    print()