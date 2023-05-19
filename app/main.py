from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
app = FastAPI()

class User(BaseModel):
    name: str
    age: int

class SPP(BaseModel):
    created_time: str
    message_tags: str
    msg : str
    pl : int
    pg : int

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/", response_model=str)
async def predict(spp: SPP):
    result = predict_pipeline(spp.created_time, spp.message_tags, spp.msg, spp.pl, spp.pg)
    return str(result)

