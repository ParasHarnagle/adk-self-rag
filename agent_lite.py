import os
import warnings
import json
import time
import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, Content

# -------------------------------------------------------------------
# ⚙️ Initialization
# -------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

APP_NAME = "ADK Streaming example"
AGENT_ENGINE_ID = "adk_self_rag_agent_lite_llm"

app = FastAPI(title="ADK Self-RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# 🔑 LLM + Voice Agent Setup
# -------------------------------------------------------------------

LITELLM_PROXY_API_KEY = os.getenv(
    "LITELLM_PROXY_API_KEY"
)

litellm.use_litellm_proxy = True

# model_name = "google/gemini-2.5-flash-lite"
model_name = "deepseek/deepseek-chat-v3.1"
model = LiteLlm(
    model=model_name,
    api_key=LITELLM_PROXY_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

# Google Search Agent
voice_agent = Agent(
    name="google_search_agent",
    model=model,
    description="Agent that answers questions using Google Search.",
    instruction="your name is pooja if someone asks how are you , reply in punjabi",
    # tools=[google_search],
)

# -------------------------------------------------------------------
# 🧠 Agent Runner Setup
# -------------------------------------------------------------------

rag_runner = InMemoryRunner(agent=voice_agent, app_name=AGENT_ENGINE_ID)


async def initialize_rag_session(user_id: str):
    """Create or reinitialize a session for the RAG agent."""
    try:
        return await rag_runner.session_service.create_session(app_name=AGENT_ENGINE_ID, user_id=user_id)
    except Exception as e:
        print("[WARN] Session init error:", e)
        return await rag_runner.session_service.create_session(app_name=AGENT_ENGINE_ID, user_id=user_id)


async def run_rag_agent_prompt(prompt: str, user_id: str):
    """Send a single prompt to the RAG agent and collect its streamed response."""
    session = await initialize_rag_session(user_id)
    content = Content(role="user", parts=[Part.from_text(text=prompt)])

    response_text = ""

    try:
        # ✅ FIXED CALL
        async for event in rag_runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content
        ):
            try:
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text + " "
            except Exception:
                pass
    except Exception as e:
        print("[ERROR] RAG agent error:", e)
        response_text = f"RAG agent encountered an error: {str(e)}"

    response_text = response_text.strip()
    print(f"[RETURN] Returning {len(response_text)} chars: {response_text[:200]}...")
    return response_text if response_text else "No response from RAG agent"

# -------------------------------------------------------------------
# 📡 FastAPI Route
# -------------------------------------------------------------------

class RAGPrompt(BaseModel):
    prompt: str
    user_id: str


@app.post("/run")
async def turbo_rag_query_endpoint(data: RAGPrompt):
    """
    POST /run
    Example:
    curl -X POST "http://127.0.0.1:8000/run" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What is FAISS?", "user_id": "user123"}'
    """
    start_time = time.time()
    print(f"[API] /run called with prompt: {data.prompt}")

    try:
        response_text = await run_rag_agent_prompt(data.prompt, data.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Agent error: {str(e)}")

    duration = round(time.time() - start_time, 2)

    clean_response = (
        response_text.replace("```json", "")
        .replace("```", "")
        .replace("{}", "")
        .strip()
    )

    try:
        parsed_response = json.loads(clean_response)
    except json.JSONDecodeError:
        parsed_response = {"text": clean_response}

    print(f"[RAG AGENT] Response ready ({duration}s): {clean_response[:120]}...")

    return {
        "prompt": data.prompt,
        "agent_response": parsed_response,
        "execution_time_sec": duration,
    }


# ---------------------- LOCAL TEST ----------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
