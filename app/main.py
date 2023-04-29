from models import manager
import threading
import asyncio

import os
from fastapi import FastAPI


app = FastAPI()

# Import endpoint definitions
from api.routes.reset import router as reset_router
from api.routes.openai import router as openai_router
from api.routes.rwkv import router as rwkv_router
app.include_router(reset_router,tags=["v1.completions.reset"])
app.include_router(openai_router,tags=["v1.engines"])
app.include_router(rwkv_router,tags=["v1.completions.rwkv"])


if __name__ == "__main__":
    # Start the models
    for model in manager.models:
        model.start()

    # Import the event listener module
    from api.listeners.event_listener import run_event_listener
    thread = threading.Thread(target=asyncio.run, args=(run_event_listener(manager),))
    thread.start()

    # Start the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)