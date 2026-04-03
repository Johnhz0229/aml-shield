"""AML-Shield FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze, cases, health
from api.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AML-Shield API started")
    init_db()
    yield


app = FastAPI(
    title="AML-Shield API",
    version="1.0.0",
    description="AI-powered Anti-Money Laundering compliance agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api/v1")
app.include_router(cases.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
