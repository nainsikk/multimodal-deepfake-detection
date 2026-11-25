
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routers import analyze
from app.routers import history
from app.routers import results
from app.routers import reports
from app.routers import sanity
from app.services.mock_predictor import MockPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Load model only once
    app.state.predictor = MockPredictor()

    logging.info("Mock prediction model loaded.")

    yield

    logging.info("Application shutting down.")


app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="Multimodal NLP Deepfake Detection and "
                "Explainable Content Verification",
    version="1.0.0",
    lifespan=lifespan
)


# Enable frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Serve static heatmaps
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)
# Serve generated explainability heatmaps
app.mount(
    "/results",
    StaticFiles(directory="results"),
    name="results"
)



# Register API router
app.include_router(analyze.router)
app.include_router(history.router)
app.include_router(results.router)
app.include_router(reports.router)
app.include_router(sanity.router)


@app.get("/")
def root():

    return {
        "message": "Multimodal Deepfake Detection API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "predictor")
    }
