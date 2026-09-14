"""FastAPI backend: /ask streams agent graph node transitions over SSE so
the frontend's trace pane can show retrieval/grading/generation live,
/health is a plain liveness check.
"""

import asyncio
import json
import os
import queue
import threading
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.graph.build import build_graph
from core.graph.state import AgentState
from core.llm.fake import FakeLLMProvider
from core.llm.groq import GroqLLMProvider
from core.providers.fake import FakeEmbeddingProvider
from core.providers.huggingface import HuggingFaceEmbeddingProvider
from core.stores.pgvector import PgVectorStore

load_dotenv()

DSN = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"

app = FastAPI(title="clinical-rag-agent API")

app.add_middleware(
    CORSMiddleware,
    # Vite picks the next free port (5174, 5175, ...) if 5173 is taken by
    # a leftover dev server, so match any localhost port rather than
    # hardcoding one and silently breaking CORS when that happens.
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


def get_embedding_provider():
    hf_api_key = os.environ.get("HF_API_KEY")
    if hf_api_key:
        hf_model = os.environ.get(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddingProvider(api_key=hf_api_key, model=hf_model)
    return FakeEmbeddingProvider(dimension=16)


def get_llm_provider():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        groq_model = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
        return GroqLLMProvider(api_key=groq_api_key, model=groq_model)
    return FakeLLMProvider()


def serialize_node_update(node_name: str, state_update: dict) -> dict:
    """Turn one LangGraph stream event into a JSON-safe trace entry.

    state_update is the AgentState field dict LangGraph yields per node;
    it contains a set (seen_chunk_ids) and SearchResult dataclasses, so it
    needs converting before json.dumps can handle it.
    """
    retrieved = state_update.get("retrieved", [])
    return {
        "node": node_name,
        "iteration": state_update.get("iteration", 0),
        "verdict": state_update.get("verdict", ""),
        "sub_queries": state_update.get("sub_queries", []),
        "retrieved_count": len(retrieved),
        "retrieved": [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section_path": r.section_path,
                "source_url": r.source_url,
                "score": r.score,
            }
            for r in retrieved
        ],
        "answer": state_update.get("answer", ""),
        "citations": state_update.get("citations", []),
    }


_SENTINEL_DONE = object()
_SENTINEL_ERROR = object()


def _run_graph_in_thread(question: str, result_queue: queue.Queue) -> None:
    """Runs the fully synchronous graph (blocking psycopg/httpx calls under
    the hood) on a worker thread, pushing each node update into a queue the
    async generator below can drain without blocking FastAPI's event loop."""
    embedding_provider = get_embedding_provider()
    vector_store = PgVectorStore(DSN, dimension=embedding_provider.dimension)
    llm_provider = get_llm_provider()

    graph = build_graph(
        llm_provider=llm_provider,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    try:
        for update in graph.stream(AgentState(question=question), stream_mode="updates"):
            for node_name, state_update in update.items():
                result_queue.put(serialize_node_update(node_name, state_update))
        result_queue.put(_SENTINEL_DONE)
    except Exception as e:
        result_queue.put((_SENTINEL_ERROR, str(e)))
    finally:
        vector_store.close()


async def stream_agent_trace(question: str) -> AsyncIterator[str]:
    result_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_run_graph_in_thread, args=(question, result_queue))
    thread.start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is _SENTINEL_DONE:
                yield "event: done\ndata: {}\n\n"
                break
            if isinstance(item, tuple) and item[0] is _SENTINEL_ERROR:
                yield f"event: error\ndata: {json.dumps({'message': item[1]})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        thread.join(timeout=5)


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_trace(request.question),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
