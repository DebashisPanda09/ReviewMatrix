import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(
    model=os.getenv("NIM_MODEL"),
    api_key=os.getenv("NIM_API_KEY"),
    base_url=os.getenv("NIM_BASE_URL")
)

response = llm.invoke("Say hello in one sentence.")
print(response.content)