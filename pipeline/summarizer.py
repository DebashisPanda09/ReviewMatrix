"""
ReviewMatrix - Summarizer (subprocess-isolated)
----------------------------------------------------
Calls aggregate_runner.py as a SEPARATE PROCESS to fetch aspect-sentiment
stats (isolating psycopg2 from this process), then uses LangChain + NVIDIA
NIM (DeepSeek) in THIS process to generate a natural-language summary.

This avoids a native DLL conflict between psycopg2 and langchain-nvidia's
aiohttp dependency that causes a crash (exit code -1073741819) when both
are active in the same Windows process.
"""

import os
import subprocess
import sys

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatNVIDIA(
    model=os.getenv("NIM_MODEL"),
    api_key=os.getenv("NIM_API_KEY"),
    base_url=os.getenv("NIM_BASE_URL"),
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    "You are an assistant that helps product managers understand customer feedback.\n"
    "Given the following aspect-level sentiment breakdown for a product:\n\n"
    "{data}\n\n"
    "Write a concise 3-4 sentence executive summary. Highlight the strongest "
    "aspect, the weakest aspect, and any notable pattern. Do not repeat raw "
    "percentages verbatim, interpret them in plain business language."
)


def run_aggregate_subprocess(*args) -> str:
    """Runs aggregate_runner.py as a subprocess and returns its stdout output."""
    result = subprocess.run(
        [sys.executable, "aggregate_runner.py", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aggregate_runner.py failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout.strip()


def get_top_product_id() -> str:
    return run_aggregate_subprocess("--top")


def get_formatted_stats(product_id: str) -> str:
    return run_aggregate_subprocess(product_id)


def generate_summary(product_id: str) -> str:
    """Generates an LLM-written executive summary for a given product_id."""
    stats_text = get_formatted_stats(product_id)
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"data": stats_text})
    return response.content


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Called with a specific product_id (e.g. from the dashboard).
        # Print ONLY the summary text to stdout.
        pid = sys.argv[1]
        summary = generate_summary(pid)
        print(summary)
    else:
        # No argument: run the verbose self-test using the top product.
        print("Fetching top product (via subprocess)...")
        pid = get_top_product_id()
        print(f"Top product: {pid}\n")

        print("Fetching formatted stats (via subprocess)...")
        stats = get_formatted_stats(pid)
        print("Formatted stats:")
        print(stats)

        print("\nCalling LLM...")
        summary = generate_summary(pid)
        print("\nSummary result:")
        print(summary)
