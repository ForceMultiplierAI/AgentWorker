import time
from typing import List, Union, Optional, Union, Dict
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic.fields import Field
from models import manager
import threading
import json
import pprint
import sseclient
import asyncio
import requests

from fastapi import FastAPI, HTTPException
from fastapi import APIRouter

router = APIRouter()

from ..models.openai import OpenAIMessage

from pydantic import BaseModel

# #  GET for /v1/completions/reset
# @router.get('/v1/completions/reset')
# async def v1_completions_reset():
#     model = manager.get_default_model()
#     model.reset()
#     return JSONResponse(content={'data': 'reset'})
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = Field(None, maxLength=64)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

@router.post('/v1/chat/completions')
async def v1_chat_completions(request: ChatCompletionRequest):
    # model = request.model
    # messages = request.messages
    # temperature = request.temperature
    # Get all the fields from the request schema, into kwargs
    kwargs = request.dict()
    # ... other fields from the request schema
    # Process the request using the manager and get the response
    model = manager.get_default_model()
    # call this def chat_complete(self, prompt : str, **kwargs):
    response = model.chat_complete(**kwargs)

    ## TODO GraphQL POST results chat_completion_audit_log chatCompletionAuditLog

    return response


@router.get('/v1/engines')
async def v1_engines():
    data = [
        {
            'object': 'engine',
            'id': id,
            'ready': True,
            'owner': 'openai',
            'permissions': None,
            'created': None
        } for id in manager.models.keys()
    ]
    return JSONResponse(content={'data': data})

# #  GET for /v1/completions/reset
# @router.get('/v1/completions/reset')
# async def v1_completions_reset():
#     model = manager.get_default_model()
#     model.reset()
#     return JSONResponse(content={'data': 'reset'})

@router.post('/v1/engines/{model_name}/completions')
async def engine_completion(message: OpenAIMessage, model_name: str = Path("RWKV-4-Pile-7B",description="Default value: RWKV-4-Pile-7B")):
    model = manager.get_named_model(model_name)
    if isinstance(message.prompt, str):
        prompt = message.prompt
    else:
        prompt = message.prompt[0]

    # length = message.length
    temperature = message.temperature
    max_tokens = message.max_tokens
    # completions = model.completions(prompt, length=length, temperature=temperature, max_tokens=max_tokens)

    # completions = manager.get_default_model()
    # self, state=None, temp: float = 1.0, top_p_usual: float = 0.8, number=1, stopStrings: List[str] = ["<|endoftext|>"], stopTokens: List[int] = [0], progressLambda=lambda args: args, end_adj=0.0
    completions = model.completions(
        prompt, 
        number=message.max_tokens,
        temp=temperature, 
        top_p_usual=message.top_p,
        stopStrings=message.stop, 
        # stopTokens=None, 
        # progressLambda=None, 
        end_adj=message.end_adj
        )
    return completions


@router.post('/v1/completions')
async def engine_completion(
        message: OpenAIMessage):
    # model = manager.get_named_model(model_name)
    model = manager.get_default_model()
    if isinstance(message.prompt, str):
        prompt = message.prompt
    else:
        prompt = message.prompt[0]
    # length = message.length
    temperature = message.temperature
    max_tokens = message.max_tokens
    # completions = model.completions(prompt, length=length, temperature=temperature, max_tokens=max_tokens)

    # completions = manager.get_default_model()
    # self, state=None, temp: float = 1.0, top_p_usual: float = 0.8, number=1, stopStrings: List[str] = ["<|endoftext|>"], stopTokens: List[int] = [0], progressLambda=lambda args: args, end_adj=0.0
    completions = model.completions(
        prompt, 
        number=message.max_tokens,
        temp=temperature, 
        top_p_usual=message.top_p, 
        stopStrings=message.stop, 
        # stopTokens=None, 
        # progressLambda=None, 
        end_adj=message.end_adj
        )
    return completions

class EmbeddingRequest(BaseModel):
    input: List[str]

@router.post('/v1/embeddings')
async def v1_embeddings(embedding_request: EmbeddingRequest):
    model = manager.get_default_model()
    return model.get_embeddings(embedding_request.input)
    
# async def generate(message: OpenAIMessage):
#     # run the model and get the output(generated text)
#     prompt = message.prompt[0]
#     temperature = float(message.temperature)
#     length = int(message.max_tokens)
#     top_p = float(message.top_p)
#     frequency_penalty = float(message.frequency_penalty)
#     presence_penalty = float(message.presence_penalty)
#     # print all above
#     print("prompt: ", prompt)
#     print("temperature: ", temperature)
#     print("length: ", length)
#     print("top_p: ", top_p)
#     print("frequency_penalty: ", frequency_penalty)
#     print("presence_penalty: ", presence_penalty)
