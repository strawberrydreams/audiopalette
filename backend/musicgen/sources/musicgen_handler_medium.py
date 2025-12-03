# MusicGen Medium 모델의 Handler 파일 (클라우드)

# transformers - Hugging Face에서 받은 로컬 MusicGen 모델
# audiocraft - Meta Flatforms에서 게시한 원본 MusicGen 모델
import logging
import re
import numpy as np
import scipy.io.wavfile
import torch

# torch._C._get_default_device() → str ("cpu" 또는 "cuda" 반환)
# 1) 공통 코어: C-API가 주는 문자열 → torch.device 객체로 감싸기
def _default_device_from_c() -> torch.device:
    return torch.device(torch._C._get_default_device())

# 2) transformers가 찾는 공개 API shim
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = _default_device_from_c # type: ignore[attr-defined]

# 3) Audiocraft가 찾는 과거 private API shim
if not hasattr(torch, '_get_default_device'):
    torch._get_default_device = _default_device_from_c # type: ignore[attr-defined]

# 4) 혹시 set_* 도 없으면 같이 만들어 두기 (필수는 아님)
if not hasattr(torch, 'set_default_device'):
    def _set_default_device(dev) -> None:
        torch._C._set_default_device(str(dev))
    torch.set_default_device = _set_default_device # type: ignore[attr-defined]

from audiocraft.models import MusicGen
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# logging 구성
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 전역 상수
# MusicGen 샘플율 권장값은 24kHz (약간 늘어짐. 한 1.3배정도 길어짐.)
# MusicGen 원본 출력값은 32kHz
MODEL_NAME = "medium"
SAMPLE_RATE = 32_000 # MusicGen 원본 샘플율 권장값
OUTPUT_BASE_DIR = Path("musicgen-outputs") / MODEL_NAME

# 기본 파라미터 (결과물 생성 기본 조건)
DEFAULT_GEN_PARAMS: Dict[str, Any] = {
    "duration": 10, # 초 단위 길이 (prompt마다 덮어쓸 수 있음)
    "top_p": 0.9,
    "temperature": 1.0,
    "top_k": 250,
    "cfg_coef": 3.0, # classifier-free guidance 계수
}

# 모델을 실행할 장치 결정 (CUDA or CPU)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE: Optional[MusicGen] = None

# 파일 실행 시 한 번만 호출되어 MusicGen Medium 모델을 device에 올려 둠
# Args: device: 명시적 torch.device 지정. None이면 _DEVICE_DEFAULT 사용.
# Returns: 캐싱된 MusicGen 인스턴스를 반환함
def load_musicgen_medium(device: Optional[torch.device] = None) -> MusicGen:
    global _MODEL_CACHE
    dev = device or _DEFAULT_DEVICE
    if _MODEL_CACHE is None:
        logger.info(f"Loading MusicGen '{MODEL_NAME}' model to {dev}")
        _MODEL_CACHE = MusicGen.get_pretrained(
            "facebook/musicgen-medium",
            device=str(dev) # cpu 또는 cuda 반환
        )
        logger.info("Model loaded successfully.")
    return _MODEL_CACHE

# 주어진 프롬프트를 이용해서 충돌을 방지하는 안전한 파일 이름을 생성함
# prompt(str) = 파일 이름 지정에 사용되는 텍스트 프롬프트
# timestamp = 날짜, 시간 형식
# 안전한 파일 이름 + .wav 형식의 파일을 반환함
def safe_filename_from_prompt(prompt: str, timestamp_format: str = "%y%m%d_%H%M%S") -> str:
    # 영문자, 숫자, 공백, 하이픈(-)이 아닌 문자를 모두 제거함
    prompt_clean = re.sub(r"[^\w\s-]", "", prompt)
    snippet = "_".join(prompt_clean.split())[:20]
    timestamp = datetime.now().strftime(timestamp_format)
    return f"{timestamp}_{MODEL_NAME}_{snippet}.wav"

# 입력받은 프롬프트를 기반으로 MusicGen Medium 모델을 구동함
# prompt: 결과물 생성에 사용되는 텍스트 프롬프트(필수)
# gen_params: duration/top_p/temperature/... 사용자가 입력한 내용을 덮어쓰면 됨
# device: 강제로 다른 디바이스에 올리고 싶을 때 사용
# returns: 생성된 .wav 파일의 경로 문자열
def generate_musicgen_medium_wav(
    prompt: str, 
    *,
    gen_params: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> str:
    if not prompt or not prompt.strip():
        # 오류 ValueError: 프롬프트가 유효하지 않음
        raise ValueError("Prompt must be a non-empty string.")
    
    model = load_musicgen_medium(device)

    # 기본 파라미터에 사용자 입력값 덮어쓰기
    _params = {**DEFAULT_GEN_PARAMS, **(gen_params or {})}
    logger.info(f"[{MODEL_NAME.upper()}] Generating audio with prompt: {_params}")

    model.set_generation_params(
        duration=_params["duration"],
        top_p=_params["top_p"],
        temperature=_params["temperature"],
        top_k=_params["top_k"],
        cfg_coef=_params["cfg_coef"],
    )

    # 결과물 생성
    # Audiocraft는 prompt를 리스트로 받음
    with torch.no_grad():
        wav_list = model.generate([prompt])
        wav_tensor = wav_list[0] # (channels, samples) 형태 float32, -1~1

    # .wav 파일 후처리 부분
    wav_array = np.clip(wav_tensor.cpu().numpy(), -1.0, 1.0) # 값 범위 [-1,1]
    wav_array = wav_array.T # (samples, channels)
    wav_int16 = (wav_array * 32767).astype(np.int16) # float -> int16

    # 생성된 .wav 파일 저장
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_BASE_DIR / safe_filename_from_prompt(prompt)
    scipy.io.wavfile.write(str(output_path), rate=SAMPLE_RATE, data=wav_int16)
    logger.info("Audio saved to %s", output_path)

    return str(output_path)

# 직접 실행 시
if __name__ == "__main__":
    TEST_PROMPT = "dreamy lo-fi beats with vinyl crackle"
    path = generate_musicgen_medium_wav(TEST_PROMPT, gen_params={"duration": 8})
    print("Generated:", path)
