# Python 인터프리터 -> 3.8.10 ('magenta-env': venv)로 바꾸기
# import MusicGEN
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile

import os
from datetime import datetime

# CUDA를 사용 가능한지 확인 (중요)
print("CUDA available:", torch.cuda.is_available())  # → True
print("Torch version:", torch.__version__)           # → 2.1.0+cu118
print("CUDA version:", torch.version.cuda)           # → 11.8
print("Device name:", torch.cuda.get_device_name(0)) # → NVIDIA RTX 3060

# 경로 설정
OUTPUT_DIR = "static/generated/musicgen/musicgen_small"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 폴더 없으면 생성

# 자연어 프롬프트 설정
prompt = "A bright and cheerful ukulele tune with claps and bells"

# 날짜 및 시간 문자열 생성 (예: 250511_2030)
timestamp = datetime.now().strftime("%y%m%d_%H%M")

# 프롬프트 일부를 파일 이름에 반영 (공백 제거 + 최대 20자)
safe_prompt = "_".join(prompt.strip().split())[:20]

# 파일명 생성
filename = f"{timestamp}_{safe_prompt}.wav"
output_path = os.path.join(OUTPUT_DIR, filename)

# 모델 로딩 (처음엔 다소 시간 걸림)
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
)

# GPU 사용 (가능하면)
if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 음악 생성
audio_values = model.generate(
    **inputs,
    do_sample=True,
    guidance_scale=3,
    max_new_tokens=256 # 대략 10초 분량
)

# WAV로 저장
audio_array = audio_values[0, 0].cpu().numpy()
scipy.io.wavfile.write(output_path, rate=16000, data=audio_array)
print(f"✅ WAV 파일 생성 완료: {output_path}")
