from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.exceptions import NewsClassifierException
import sys

app = FastAPI(title="News Categorization")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    try:
        pipeline = PredictionPipeline()
        result = pipeline.predict(text)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "text": text
            }
        )
    except Exception as e:
        raise NewsClassifierException(e, sys)
