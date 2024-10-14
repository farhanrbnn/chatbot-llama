from fastapi import FastAPI
from src.llm.delivery.llm import llm_api

app = FastAPI()

app.include_router(llm_api)

