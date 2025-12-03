# Paperspace 가상머신 환경에서 CUDA가 제대로 인식되는지 확인하는 테스트 파일
import torch, platform, subprocess, os
print("torch            :", torch.__version__)
print("CUDA runtime     :", torch.version.cuda)
print("is_available     :", torch.cuda.is_available()) # CUDA 인식 성공하면 여기가 True
print("device_count     :", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
