# Magenta Basic 핸들러 (로컬)
# Requirements:
#   magenta==2.1.0
#   note_seq==0.0.3

import logging, note_seq, os, re
from datetime import datetime
from pathlib import Path
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from prompt_to_midi_params import prompt_to_params

# 초기 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 번들 경로 설정
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model" / "basic_rnn.mag"

# 번들 및 제너레이터 전역 초기화
try:
    bundle = sequence_generator_bundle.read_bundle_file(str(BUNDLE_PATH))
    generator = melody_rnn_sequence_generator.get_generator_map()['basic_rnn'](
        checkpoint=None, bundle=bundle
    )
    generator.initialize()
    logger.info("Magenta Basic RNN 모델이 성공적으로 초기화되었습니다.")
except Exception as e:
    logger.error(f"Magenta 모델 초기화 실패: {e}")
    raise RuntimeError(f"Magenta 모델 초기화에 실패했습니다: {e}")


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
    safe = re.sub(r"[^0-9a-zA-Z가-힣_]", "_", text.strip())
    return safe[:max_length]


def generate_magenta_basic_midi(
    prompt: str,
    qpm: float = 120.0,
    duration_secs: float = 4.0
) -> str:
    """
    Magenta Basic RNN 모델을 사용해 시퀀스를 생성하고 MIDI 파일로만 저장합니다.

    Args:
        prompt (str): 텍스트 프롬프트(파일명 생성에 사용되며, 실제 시드 노트 생성에는 ChatGPT API를 통해 변환됨).
        qpm (float): 시퀀스 템포(Quarter Notes Per Minute).
        duration_secs (float): 시드 이후 생성될 음악 길이(초).

    Returns:
        str: 생성된 MIDI 파일 경로.
    """
    # 1) ChatGPT API를 통해 prompt → 시드 노트 리스트(JSON) 변환
    try:
        notes_params = prompt_to_params(prompt)
        logger.info(f"ChatGPT를 통해 변환된 시드 노트 리스트: {notes_params}")
    except Exception as e:
        logger.error(f"prompt_to_params 호출 실패: {e}")
        raise

    # 2) NoteSequence 생성
    seed = note_seq.NoteSequence()
    seed.total_time = 0.0
    seed.tempos.add(qpm=qpm)

    time_cursor = 0.0
    for note_dict in notes_params:
        pitch = note_dict["pitch"]
        dur = float(note_dict["duration"])
        seed.notes.add(
            pitch=pitch,
            start_time=time_cursor,
            end_time=time_cursor + dur,
            velocity=80
        )
        time_cursor += dur

    # 빈 시드일 때 기본 노트 한 개 추가
    if len(seed.notes) == 0:
        seed.notes.add(pitch=60, start_time=0.0, end_time=0.5, velocity=80)
        seed.total_time = 0.5
    else:
        seed.total_time = time_cursor

    # 3) Magenta 생성 옵션 구성
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.generate_sections.add(
        start_time=seed.total_time,
        end_time=seed.total_time + duration_secs
    )

    # 시퀀스 생성
    try:
        sequence = generator.generate(seed, generator_options)
        logger.info("시퀀스 생성 완료.")
    except Exception as e:
        logger.error(f"시퀀스 생성 실패: {e}")
        raise RuntimeError(f"시퀀스 생성에 실패했습니다: {e}")

    # 4) MIDI 파일 저장
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")  # ISO 8601 형식
    safe_prompt = sanitize_filename(prompt)
    filename_base = f"{timestamp}_{safe_prompt}_magenta_basic"
    output_dir = BASE_DIR / "magenta-outputs" / "basic_rnn"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"출력 디렉터리 생성 실패: {output_dir} - {e}")
        raise RuntimeError(f"출력 디렉터리 생성 실패: {e}")

    midi_path = output_dir / f"{filename_base}.mid"
    try:
        note_seq.sequence_proto_to_midi_file(sequence, str(midi_path))
        logger.info(f"MIDI 파일 저장: {midi_path}")
    except Exception as e:
        logger.error(f"MIDI 파일 저장 실패: {e}")
        raise RuntimeError(f"MIDI 파일 저장에 실패했습니다: {e}")

    return str(midi_path)
