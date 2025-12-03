# magenta_handler_drums.py
# Requirements:
#   magenta==2.1.0
#   note_seq==0.0.3
#   openai==0.27.0
#   python-dotenv==1.0.0 (선택)

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import note_seq
from note_seq import sequences_lib
from magenta.models.drums_rnn import drums_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

# ChatGPT API를 통한 prompt → 노트 리스트 변환 함수
from prompt_to_midi_params import prompt_to_params

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 번들(.mag) 경로
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model" / "drum_kit_rnn.mag"  # 실제 파일명 확인 후 수정

# 1) 번들 로드
try:
    bundle = sequence_generator_bundle.read_bundle_file(str(BUNDLE_PATH))
    logger.info("드럼 번들 파일 로드 성공")
except Exception as e:
    logger.error(f"드럼 번들 파일 로드 실패: {e}")
    raise RuntimeError(f"Magenta 드럼 모델 번들 로드 실패: {e}")

# 2) 제너레이터 초기화
try:
    drum_map = drums_rnn_sequence_generator.get_generator_map()
    if 'drum_kit' not in drum_map:
        logger.error(f"가능한 드럼 키: {list(drum_map.keys())}")
        raise KeyError("`drum_kit` 키를 찾을 수 없습니다")
    generator = drum_map['drum_kit'](checkpoint=None, bundle=bundle)
    generator.initialize()
    logger.info("Magenta Drum Kit RNN 모델 초기화 성공")
except Exception as e:
    logger.error(f"Magenta 드럼 모델 초기화 실패: {e}")
    raise RuntimeError(f"Magenta 드럼 모델 초기화 실패: {e}")


def sanitize_filename(text: str, max_length: int = 20) -> str:
    """
    파일명으로 사용 가능한 문자열로 변환합니다.
    영숫자, 한글, 언더스코어만 남기고 나머지는 언더스코어로 대체합니다.

    Args:
        text (str): 원본 문자열.
        max_length (int): 잘라낼 최대 문자 수.

    Returns:
        str: 파일명 안전 문자열.
    """
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in text.strip())
    return safe[:max_length]


def generate_magenta_drums_midi(
    prompt: str,
    qpm: float = 120.0,
    duration_secs: float = 4.0
) -> str:
    """
    드럼 RNN을 사용해 시드 NoteSequence를 ChatGPT API로부터 받아 생성하고,
    결과를 MIDI 파일로 저장합니다.

    Args:
        prompt (str): 텍스트 설명 (ChatGPT로 변환하여 노트 리스트 생성).
        qpm (float): 시드 템포 (Quarter Notes Per Minute).
        duration_secs (float): 시드 이후 생성될 음악 길이 (초).

    Returns:
        str: 생성된 MIDI 파일 경로.
    """
    # 1) ChatGPT API를 통해 prompt → 드럼 노트 리스트(Json) 변환
    try:
        notes_params = prompt_to_params(prompt)
        logger.info(f"ChatGPT를 통해 변환된 드럼 노트 리스트: {notes_params}")
    except Exception as e:
        logger.error(f"prompt_to_params 호출 실패: {e}")
        raise

    # 2) NoteSequence 객체로 시드 노트 삽입
    seed = note_seq.NoteSequence()
    seed.total_time = 0.0
    seed.tempos.add(qpm=qpm)
    time_cursor = 0.0

    for note_dict in notes_params:
        pitch = note_dict.get("pitch")
        dur = float(note_dict.get("duration", 0.5))
        # 드럼 프로그램(pgm) 설정: 드럼 킷 예제에서는 이미 General MIDI 드럼 채널 가정
        seed.notes.add(
            pitch=pitch,
            start_time=time_cursor,
            end_time=time_cursor + dur,
            velocity=100,
            is_drum=True
        )
        time_cursor += dur

    # 시드 노트가 하나도 없다면 기본 킥 한 개 삽입
    if len(seed.notes) == 0:
        seed.notes.add(pitch=36, start_time=0.0, end_time=0.5, velocity=100, is_drum=True)
        seed.total_time = 0.5
    else:
        seed.total_time = time_cursor

    # 3) Magenta 생성 옵션 구성
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.generate_sections.add(
        start_time=seed.total_time,
        end_time=seed.total_time + duration_secs
    )

    # 4) 시퀀스 생성
    try:
        sequence = generator.generate(seed, generator_options)
        logger.info("드럼 시퀀스 생성 완료")
    except Exception as e:
        logger.error(f"드럼 시퀀스 생성 실패: {e}")
        raise RuntimeError(f"드럼 시퀀스 생성에 실패했습니다: {e}")

    # 5) 파일명 및 경로 생성
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_prompt = sanitize_filename(prompt)
    filename_base = f"{timestamp}_{safe_prompt}_drum"
    output_dir = BASE_DIR / "magenta-outputs" / "drum_rnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_path = output_dir / f"{filename_base}.mid"
    try:
        note_seq.sequence_proto_to_midi_file(sequence, str(midi_path))
        logger.info(f"드럼 MIDI 파일 저장: {midi_path}")
    except Exception as e:
        logger.error(f"드럼 MIDI 파일 저장 실패: {e}")
        raise RuntimeError(f"드럼 MIDI 파일 저장에 실패했습니다: {e}")

    return str(midi_path)
