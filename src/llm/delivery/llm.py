from fastapi import APIRouter 
from typing import Dict
from fastapi.encoders import jsonable_encoder

from src.llm.usecase.llm import LLMusecase

llm_usecase = LLMusecase()

llm_api = APIRouter()

@llm_api.post("/api/v1/generate")
def generate(request: Dict):
    response = llm_usecase.generate(request)

    return jsonable_encoder(response)
