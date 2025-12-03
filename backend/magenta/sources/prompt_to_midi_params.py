# prompt_to_midi_params.py
# Requirements:
#   openai>=1.0.0
#   python-dotenv>=1.0.0

import os
import json
import logging
from typing import List, Dict
from pathlib import Path

import openai  # 모듈 레벨 방식 사용
from dotenv import load_dotenv

# 로깅 레벨 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env(또는 OPENAI_API_KEY.env) 파일에서 API 키 불러오기
env_path = Path(__file__).resolve().parent / "OPENAI_API_KEY.env"
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logger.error("환경 변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

# system / few-shot 예시 메시지
system_message = {
    "role": "system",
    "content": (
        "당신은 사용자가 입력한 멜로디 설명을 { \"pitch\": 정수, \"duration\": 실수 } 꼴의 JSON 배열로만 반환하는 역할을 담당합니다. "
        "절대로 추가 설명이나 자연어 텍스트를 덧붙이지 말고, 오직 순수 JSON 배열만 반환하세요. "
        "JSON 이외의 텍스트, 주석, 코드블록 구분 기호(```), 설명문 등을 일절 포함하지 않습니다."
    )
}

few_shot_user = {
    "role": "user",
    "content": (
        "다음 사용자 예시들을 참고하여, 요청된 각 멜로디 설명을 JSON 배열로 변환하세요.\n\n"
        "예시 1:\n"
        "사용자: \"단순한 C 메이저 스케일 (도-레-미-파-솔), 각 음마다 0.5초씩\"\n"
        "어시스턴트: [{\"pitch\":60,\"duration\":0.5},{\"pitch\":62,\"duration\":0.5},{\"pitch\":64,\"duration\":0.5},{\"pitch\":65,\"duration\":0.5},{\"pitch\":67,\"duration\":0.5}]\n\n"
        "예시 2:\n"
        "사용자: \"느리고 감성적인 발라드: A3(57) 두 번, C4(60) 두 번, E4(64) 한 번, 각 음 1초\"\n"
        "어시스턴트: [{\"pitch\":57,\"duration\":1.0},{\"pitch\":57,\"duration\":1.0},{\"pitch\":60,\"duration\":1.0},{\"pitch\":60,\"duration\":1.0},{\"pitch\":64,\"duration\":1.0}]\n\n"
        "예시 3:\n"
        "사용자: \"밝고 경쾌한 팝 멜로디: C4, E4, G4, B4 음들을 0.5초씩 뛴 뒤, 마지막에 C5 1초\"\n"
        "어시스턴트: [{\"pitch\":60,\"duration\":0.5},{\"pitch\":64,\"duration\":0.5},{\"pitch\":67,\"duration\":0.5},{\"pitch\":71,\"duration\":0.5},{\"pitch\":72,\"duration\":1.0}]\n\n"
        "---\n\n"
        "이제 아래의 ‘새 사용자 요청’만 처리하세요. 예시와 똑같은 형식을 반드시 따르고, 다른 설명은 절대로 추가하지 마세요."
    )
}

def prompt_to_params(prompt: str) -> List[Dict]:
    user_message = {
        "role": "user",
        "content": prompt,
    }

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1",  # 유효한 모델 이름으로 변경
            messages=[system_message, few_shot_user, user_message],
            temperature=0.7,
            max_tokens=512,
            n=1
        )
    except Exception as e:
        logger.error(f"ChatGPT API 호출 중 오류 발생: {e}")
        raise RuntimeError(f"ChatGPT API 호출 실패: {e}")

    # 1) 원문 추출
    raw_content = response.choices[0].message.content.strip()
    logger.info(f"ChatGPT 응답 원문(가공 전): '''{raw_content}'''")

    # 2) 코드블록 제거
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`").strip()
        logger.info(f"코드블록 제거 후: '''{raw_content}'''")

    # 3) JSON 파싱
    try:
        notes = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error(f"ChatGPT 응답 JSON 파싱 실패: {e}\n원문: {raw_content}")
        raise RuntimeError(f"응답 JSON 파싱 실패: {e}")

    # 4) 기본 검증
    if not isinstance(notes, list):
        raise ValueError(f"파싱된 결과가 리스트가 아닙니다: {type(notes)}")

    for idx, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"각 항목이 dict 형태가 아닙니다: idx={idx}, value={note}")
        if "pitch" not in note or "duration" not in note:
            raise ValueError(f"각 항목에 'pitch' 또는 'duration' 필드가 없습니다: {note}")
        if not isinstance(note["pitch"], int):
            raise ValueError(f"'pitch'는 int여야 합니다: {note}")
        if not isinstance(note["duration"], (int, float)):
            raise ValueError(f"'duration'은 숫자여야 합니다: {note}")

    return notes
