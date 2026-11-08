import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NIM_API_KEY")

r = requests.get(
    "https://integrate.api.nvidia.com/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
data = r.json()
matches = [m["id"] for m in data["data"] if "llama" in m["id"].lower()]

print("Llama models found:")
for m in matches:
    print(" -", m)