from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1
    )
    yield


app = FastAPI(title="ModGuard AI", lifespan=lifespan)


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {"status": "ModGuard API running"}


@app.post("/moderate")
def moderate(data: Message):
    results = model(
        data.message,
        top_k=None,
        truncation=True,
        max_length=512
    )

    scores = {
        result["label"]: round(float(result["score"]), 4)
        for result in results
    }

    toxicity = max(scores.values())

    if toxicity >= 0.80:
        decision = "blocked"
    elif toxicity >= 0.60:
        decision = "flagged"
    else:
        decision = "approved"

    return {
        "decision": decision,
        "toxicity_score": round(toxicity, 4),
        "categories": scores
    }

