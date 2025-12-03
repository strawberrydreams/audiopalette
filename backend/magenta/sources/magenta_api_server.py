# main.py

from __future__ import annotations

import os
import shutil
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Callable
from contextlib import asynccontextmanager

import numpy as np
import pretty_midi
from scipy.io import wavfile
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# --- 로깅 구성 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("magenta_api")

# --- 모델별 핸들러 파일 불러오기 ---
try:
    from magenta_handler_basic import generate_magenta_basic_midi
    from magenta_handler_drums import generate_magenta_drums_midi
    from magenta_handler_attention import generate_magenta_attention_midi
except ImportError as e:
    logger.exception(f"Failed to import Magenta handlers: {e}")
    raise

# --- FastAPI 구성 ---
app = FastAPI(
    title="Magenta API Server",
    description="Serve Magenta-based MIDI-to-WAV generation endpoints.",
    version="1.0.0",
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

# --- 허용 가능한 모델 및 해당 핸들러 매핑 ---
# PEP 585 스타일(dict[str, Callable[[str], str]]) 대신,
# typing.Dict와 typing.Callable을 사용한 방식
VALID_MODELS: Dict[str, Callable[[str], str]] = {
    "magenta-basic": generate_magenta_basic_midi,
    "magenta-drums": generate_magenta_drums_midi,
    "magenta-attention": generate_magenta_attention_midi,
}

# 최대 프롬프트 길이 제한 (문자 수)
MAX_PROMPT_LENGTH = 1024

# --- 생성된 .wav 파일을 클라우드에서 삭제 ---
# index.html에서는 상관 없이 다운로드 가능, 나중에 기록 시스템 추가를 위해 수정 필요
def cleanup_file(path: str) -> None:
    try:
        os.remove(path)
        logger.info(f"Removed file: {path}")
    except OSError as e:
        logger.warning(f"Cleanup failed for ({path}): {e}")

# --- MIDI 파일을 WAV로 변환하여 저장 (pretty_midi와 fluidsynth 이용) --- 
# 음량이 0에 가까운 경우를 방지하기 위해 정규화를 수행
def convert_midi_to_wav(midi_path: str, wav_path: str) -> None:
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        # fluidsynth를 쓰기 위해서는 시스템에 fluidsynth와 SoundFont 설치 필요
        audio = pm.fluidsynth(fs=44100)
        peak = np.max(np.abs(audio))
        if peak < 1e-9:
            peak = 1.0
        audio_int16 = np.int16(audio / peak * 32767)
        wavfile.write(wav_path, 44100, audio_int16)
        logger.info(f"WAV 파일 생성 완료: {wav_path}")
    except Exception as e:
        logger.error(f"MIDI → WAV 변환 실패 ({midi_path}): {traceback.format_exc()}")
        raise RuntimeError(f"MIDI → WAV 변환 중 오류가 발생했습니다: {e}")

@app.get("/")
async def root():
    return {"message": "Magenta Local Proxy Server"}

# --- 토큰 시스템 기반의 /generate 엔드포인트 ---
@app.post(
    "/generate",
    summary="Generate WAV from prompt",
    description="Generate music based on the given prompt and model.",
)
async def generate(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    model: str = Form(...)
):
    # 1. 입력 검증
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="프롬프트가 비어 있습니다.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"프롬프트는 최대 {MAX_PROMPT_LENGTH}자까지 허용됩니다."
        )
    handler = VALID_MODELS.get(model)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 모델: {model}")

    # 2. 임시 디렉터리 생성 (각 요청마다 새로운 폴더)
    temp_dir = tempfile.mkdtemp(prefix="magenta_temp_")
    temp_dir_path = Path(temp_dir)
    logger.info(f"임시 디렉터리 생성: {temp_dir}")

    try:
        # 3. MIDI 생성 (블로킹 작업이므로 run_in_threadpool 사용)
        midi_path = await run_in_threadpool(handler, prompt)
        # handler가 반환하는 midi_path가 project 내 기본 디렉터리라면, 이를 임시 디렉터리로 복사
        midi_path = Path(midi_path)
        dest_midi_path = temp_dir_path / midi_path.name
        shutil.copy(str(midi_path), str(dest_midi_path))
        logger.info(f"MIDI 파일 복사: {dest_midi_path}")

        # 이후 원본 MIDI 파일은 삭제 대상으로 추가 (백그라운드에서 삭제)
        background_tasks.add_task(cleanup_file, str(midi_path))

        # 4. WAV 변환
        wav_filename = dest_midi_path.with_suffix(".wav").name
        wav_path = temp_dir_path / wav_filename
        await run_in_threadpool(convert_midi_to_wav, str(dest_midi_path), str(wav_path))

        # 5. 응답 전송 및 파일 정리 태스크 등록
        background_tasks.add_task(cleanup_file, str(dest_midi_path))
        background_tasks.add_task(cleanup_file, str(wav_path))
        background_tasks.add_task(shutil.rmtree, str(temp_dir_path))

        return FileResponse(
            path=str(wav_path),
            media_type="audio/wav",
            filename=wav_path.name
        )

    except HTTPException:
        # 이미 HTTPException을 발생시킨 경우 그대로 전달
        # 임시 디렉터리도 정리
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        # 예상치 못한 예외: 임시 디렉터리 정리 후 500 응답
        logger.error(f"/generate 처리 중 예외 발생: {traceback.format_exc()}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")
