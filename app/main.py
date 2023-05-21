from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
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


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(spp: SPP):
    result = predict_pipeline(spp.created_time, spp.message_tags, spp.msg, spp.pl, spp.pg, spp.type)
    return str(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=443, ssl_keyfile="/app/app/private_key.key", ssl_certfile="/app/app/certificate.crt")
