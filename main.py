from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline
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
    type: str

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(spp: SPP):
    result = predict_pipeline(spp.created_time, spp.message_tags, spp.msg, spp.pl, spp.pg, spp.type)
    return str(result)

