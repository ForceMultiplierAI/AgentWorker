import time
from typing import List, Union
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
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


#  GET for /v1/completions/reset
@router.get('/v1/completions/reset')
async def v1_completions_reset():
    model = manager.get_default_model()
    model.reset()
    return JSONResponse(content={'data': 'reset'})