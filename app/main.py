"""FastAPI app entry point, health check, CORS, and structured logging."""

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import evaluate

_log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_log_level),
)

app = FastAPI(
    title="M42 CPG Report Evaluation API",
    description=(
        "Evaluates LLM-generated Clinical Practice Guideline reports for "
        "clinical accuracy, safety, hallucinations, and evidence traceability "
        "using an LLM-as-judge pattern with majority voting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "healthy", "version": "1.0.0"}
