import time
# from typing import List, Union
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

from typing import List, Optional, Union, Dict
from pydantic import BaseModel


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

router = APIRouter()

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
    return response

#  GET for /v1/completions/reset
@router.get('/v1/completions/reset')
async def v1_completions_reset():
    model = manager.get_default_model()
    model.reset()
    return JSONResponse(content={'data': 'reset'})
    
    
@router.get('/v1/completions/loadStateIdemponent')
async def v1_completions_loadStateIdemponent(state_name):
    model = manager.get_default_model()
    model.loadStateIdemponent(state_name)
    return JSONResponse(content={'data': 'OK'})
    
@router.get('/v1/completions/saveState')
async def v1_completions_saveState(state_name):
    model = manager.get_default_model()
    model.saveState(state_name)
    return JSONResponse(content={'data': 'OK'})

@router.get('/v1/completions/listStates')
async def v1_completions_listStates():
    model = manager.get_default_model()
    return model.listStates()



# # POST chat.completions
# @router.post('/v1/chat/completions')
# async def v1_chat_completions(body: dict):
#     model = manager.get_default_model()
#     return model.chat_complete(body)

# @router.post('/v1/chat/completions')
# async def v1_chat_completions(body: ChatCompletionRequest):
#     model = manager.get_default_model()
#     return model.chat_complete(body.dict())



# #  GET for /v1/completions/reset
# @router.post('/v1/chat/completions')
# async def v1_chat_completions():
#     model = manager.get_default_model()
#     response = manager.chat_complete(prompt, model=model, **kwargs)
#     return JSONResponse(content={'data': 'reset'})
