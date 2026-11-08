import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NIM_API_KEY")
print("Key loaded:", bool(api_key), "- length:", len(api_key) if api_key else 0)

url = "https://integrate.api.nvidia.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

try:
    r = requests.get(url, headers=headers, timeout=30)
    print("Status code:", r.status_code)
    print("Response:", r.text[:500])
except Exception as e:
    print("Request failed:", e)

import json
data = r.json()
matches = [m["id"] for m in data["data"] if "deepseek" in m["id"].lower()]
print("\nDeepSeek models found:")
for m in matches:
    print(" -", m)