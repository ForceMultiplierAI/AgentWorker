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




router = APIRouter()



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
