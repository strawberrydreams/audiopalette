import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# 1. 디바이스 설정 (GPU 사용 가능하면 GPU로)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CUDA를 사용 가능한지 확인 (중요)
print("CUDA available:", torch.cuda.is_available())  # → True
print("Torch version:", torch.__version__)           # → 2.1.0+cu118
print("CUDA version:", torch.version.cuda)           # → 11.8
print("Device name:", torch.cuda.get_device_name(0)) # → NVIDIA RTX 3060

# 2. MusicGen 모델 불러오기 (small, medium, large 가능)
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.to(device)  # GPU로 모델 이동

# 3. 생성 파라미터 설정
model.set_generation_params(duration=30)  # 30초짜리 생성

# 4. 텍스트 프롬프트 입력
descriptions = [
    "a calm piano melody with ambient background and slow tempo"
]

# 5. 음악 생성 (batch로도 가능)
print("Generating music...")
wav = model.generate(descriptions)  # 반환값은 Tensor [B, 1, T]

# 6. 생성된 오디오 저장
audio_write("output_music", wav[0].cpu(), model.sample_rate, strategy="loudness")
print("Saved to output_music.wav")
