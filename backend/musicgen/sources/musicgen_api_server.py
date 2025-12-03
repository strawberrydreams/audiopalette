# Paperspace 가상환경 백엔드 파일
# /generate 엔드포인트에서 사용자 입력값을 받고, 메타태그가 사용되었을 경우 지정 파라미터만 덮어쓰기 방식으로 음악을 생성한다.
# - prompt (필수)
# - model  (musicgen-small / -medium / -large)
# - duration / temperature / top_p / top_k / cfg_coef (선택)

# 타입 주석을 문자열로 남겨 두는 스위치
# 반드시 파일 도입부 첫 줄에 있어야 하는 호출
from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Literal, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from sqlalchemy import Table, Column, Text, Integer, MetaData, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

""" # DB 만들고 다시 열기
# --- 데이터베이스 도메인 ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/dbname")

# --- 데이터베이스 구성 (실제 DB와 동기화) ---
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
metadata = MetaData()
user_tokens = Table(
    "user_tokens", metadata,
    Column("api_key", Text, primary_key=True),
    Column("basic_remaining", Integer, nullable=False, default=0),
    Column("advanced_remaining", Integer, nullable=False, default=0),
)
# To create tables, run metadata.create_all(bind=engine.sync_engine) in a sync context

# --- 토큰 충전/소진 ---
async def consume_token(api_key: str, model: str):
    column = (
        user_tokens.c.basic_remaining
        if model in ("musicgen-small", "musicgen-medium")
        else user_tokens.c.advanced_remaining
    )
    stmt = (
        update(user_tokens)
        .where(user_tokens.c.api_key == api_key)
        .where(column > 0)
        .values({column: column - 1})
    )
    async with AsyncSessionLocal() as session:
        try: 
            result = await session.execute(stmt)
            if result.rowcount == 0:
                await session.rollback()
                raise HTTPException(status_code=403, detail="토큰이 부족하거나 유효하지 않습니다.")
            await session.commit()
        except Exception as e:
            logging.getLogger("musicgen_token").exception(f"Error consuming token for key={api_key}: {e}")
            raise
"""
# --- 로깅 구성 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("musicgen_api")

# --- 모델별 핸들러 파일 불러오기 ---
try:
    from musicgen_handler_small import generate_musicgen_small_wav, load_musicgen_small
    from musicgen_handler_medium import generate_musicgen_medium_wav, load_musicgen_medium
    from musicgen_handler_large import generate_musicgen_large_wav, load_musicgen_large
except ImportError as e:
    logger.exception(f"Failed to import MusicGen handlers: {e}")
    raise

# --- GPU에 모델을 한 번만 로딩하는 전역 모델 캐싱 ---
small_model = None
medium_model = None
large_model = None

# --- Lifespan을 이용한 비동기 수명 주기 관리 함수 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global small_model, medium_model, large_model # 모델을 전역 변수로 선언
    logger.info("Loading MusicGen models into GPU...") # 모델을 하나씩 GPU에 올리고
    small_model = load_musicgen_small()
    medium_model = load_musicgen_medium()
    large_model = load_musicgen_large()
    logger.info("Models loaded.") # 모델 로딩 완료
    yield # 앱이 종료된 후에 실행할 작업을 지정할 수 있음
    # Optional cleanup

# --- FastAPI 구성 ---
app = FastAPI(
    title="MusicGen API Server",
    description="Serve MusicGen small, medium, and large models for WAV generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS 구성 ---
# 미리 설정된 출처에게만 CORS를 허용함
# regex를 사용하면 Localhost 또는 127.0.0.1을 메인 도메인으로 하는 출처가 모두 허용됨 (포트가 변경될 때마다 편집할 필요 없음)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[], # regex를 사용하기 위해 이 칸을 빈칸으로 변경함 (regex 대신 도메인 링크를 허용하도록 변경)
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?", # Localhost 출처를 모두 허용 (추후 삭제)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- 안전한 동시성을 위한 세마포어 구성 ---
# 현재는 한번에 1개 요청만 처리 가능, GPU 자원에 따라 1 -> N으로 늘릴 수 있음
GPU_SEMAPHORE = asyncio.Semaphore(1)

# --- 입력받은 모델명에 알맞은 함수를 호출 ---
def _sync_generate(prompt: str, model: str, gen_params: Dict[str, Any]) -> str:
    if model == "musicgen-small":
        return generate_musicgen_small_wav(prompt, gen_params=gen_params)
    if model == "musicgen-medium":
        return generate_musicgen_medium_wav(prompt, gen_params=gen_params)
    if model == "musicgen-large":
        return generate_musicgen_large_wav(prompt, gen_params=gen_params)
    raise ValueError(f"Unsupported model: {model}")

# --- 생성된 .wav 파일을 클라우드에서 삭제 ---
# index.html에서는 상관 없이 다운로드 가능, 나중에 기록 시스템 추가를 위해 수정 필요
def _cleanup_file(path: str):
    try:
        os.remove(path)
        logger.info(f"Removed file: {path}")
    except OSError as e:
        logger.warning(f"Cleanup failed for {path}: {e}")

# --- 서버가 정상 작동하는지 확인하는 함수 ---
@app.get("/", summary="Health Check")
async def root():
    return {"message": "MusicGen API Server is running."}

# --- 토큰 시스템 기반의 /generate 엔드포인트 ---
@app.post(
    "/generate",
    summary="Generate WAV from prompt",
    description="Generate music based on the given prompt and model.",
)
async def generate(
    *,
    prompt: str = Form(..., max_length=200),
    model: Literal["musicgen-small", "musicgen-medium", "musicgen-large"] = Form("musicgen-small"),
    duration: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    cfg_coef: Optional[float] = Form(None),
    # token: str = Form(..., description="API token for token consumption"),
    background_tasks: BackgroundTasks,
):
    """
    # 토큰 소진
    try:
        await consume_token(token, model)
        logger.debug(f"Token consumed for key={token}, model={model}")
    except HTTPException as e:
        logger.error(f"Token consumption error: status={e.status_code}, detail={e.detail}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during token consumption: {e}")
        raise HTTPException(status_code=500, detail="Internal token processing error")
    """

    start = time.time()
    logger.info(f"Request model={model}, prompt={prompt[:30]}...")

    # 사용자가 /generate 요청에서 넘겨준 생성 파라미터를 딕셔너리로 정리 (None이 아닌 것만)
    params = {k: v for k, v in {
        "duration": duration,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "cfg_coef": cfg_coef,
    }.items() if v is not None}

    # 실제로 GPU에 로드된 모델을 실행하고, .wav 파일을 반환하는 부분
    async with GPU_SEMAPHORE:
        try:
            output = await run_in_threadpool(_sync_generate, prompt, model, params)
            elapsed = time.time() - start # start에서 elapsed까지 몇 초가 걸렸는지 로그로 출력함
            logger.info(f"Generated in {elapsed:.2f}s: {output}")
            background_tasks.add_task(_cleanup_file, output) # _cleanup_file 함수로 파일 자동 삭제
            return FileResponse(path=output, media_type="audio/wav", filename=os.path.basename(output))
        except ValueError as ve:
            logger.error(f"Bad request: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception:
            logger.exception("Internal error during generation")
            raise HTTPException(status_code=500, detail="Internal server error")

# TODO: SSL 설정, 추가 속도 제한 정책(IP 기반 또는 토큰 버킷) 구현
# TODO: Prometheus/Grafana를 설정하여 지표(지연 시간, 오류율, GPU 사용률)를 확인합니다.
# 재현 가능한 배포를 위해 requirements.txt에 종속성을 고정합니다.
