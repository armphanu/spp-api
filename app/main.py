from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from fastapi.middleware.cors import CORSMiddleware
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


# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

origins = [
    "http://localhost",
    "http://localhost:8080",  # Replace with your frontend URL
]

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

