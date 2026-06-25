import argparse
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from llama_client_hf import LlamaHFClient


app = FastAPI()
CLIENT: Optional[LlamaHFClient] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama8b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 4096
    max_new_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = "llama8b"
    prompt: Any
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 4096
    max_new_tokens: Optional[int] = None
    stream: Optional[bool] = False


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    chunks = []
    for m in messages:
        chunks.append(f"{m.role.upper()}:\n{m.content}")
    return "\n\n".join(chunks)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "llama8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-hf",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    global CLIENT

    assert CLIENT is not None, "HF client is not initialized"

    CLIENT.temperature = req.temperature if req.temperature is not None else CLIENT.temperature
    CLIENT.top_p = req.top_p if req.top_p is not None else CLIENT.top_p

    if req.max_new_tokens is not None:
        CLIENT.max_new_tokens = req.max_new_tokens
    elif req.max_tokens is not None:
        CLIENT.max_new_tokens = req.max_tokens

    messages = [
    {
        "role": m.role,
        "content": m.content,
    }
    for m in req.messages
    ]

    output = CLIENT.generate_messages(
        messages=messages,
        max_new_tokens=req.max_new_tokens or req.max_tokens or 4096,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    return {
        "id": f"chatcmpl-local-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest) -> Dict[str, Any]:
    global CLIENT

    assert CLIENT is not None, "HF client is not initialized"

    CLIENT.temperature = req.temperature if req.temperature is not None else CLIENT.temperature
    CLIENT.top_p = req.top_p if req.top_p is not None else CLIENT.top_p

    if req.max_new_tokens is not None:
        CLIENT.max_new_tokens = req.max_new_tokens
    elif req.max_tokens is not None:
        CLIENT.max_new_tokens = req.max_tokens

    if isinstance(req.prompt, list):
        prompt = "\n\n".join(str(x) for x in req.prompt)
    else:
        prompt = str(req.prompt)

    output = CLIENT.generate_messages(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    max_new_tokens=req.max_new_tokens or req.max_tokens or 4096,
    temperature=req.temperature,
    top_p=req.top_p,
)

    return {
        "id": f"cmpl-local-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "text": output,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def main():
    global CLIENT

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    CLIENT = LlamaHFClient(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_4bit=args.use_4bit,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()