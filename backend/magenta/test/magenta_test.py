# Python 인터프리터 -> 3.8.10 ('magenta-env': venv)로 바꾸기
# import Magenta
from magenta.models.melody_rnn import melody_rnn_sequence_generator

from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle

from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import note_seq
import os
from datetime import datetime

# 경로 설정
OUTPUT_DIR = 'magenta-server/magenta-outputs/basic_rnn'
os.makedirs(OUTPUT_DIR, exist_ok=True) # 폴더 없으면 생성

# 모델 로딩 (사전 다운로드)
BUNDLE_PATH = 'magenta-server/model/basic_rnn.mag'
MODEL_NAME = 'basic_rnn'

# 파라미터 설정
NUM_OUTPUTS = 5 # 5개 파일 생성
NUM_STEPS = 128 # 128은 16분음표 기준 8마디 길이
PRIMER_MELODY = [60]  # '도' 음으로 시작

def main():
    # 모델 로드
    print("Loading model...")

    # 로딩된 모델을 가져와서 음악 생성
    # bundle = note_seq.read_bundle_file(BUNDLE_PATH)
    bundle = sequence_generator_bundle.read_bundle_file(BUNDLE_PATH)
    generator_map = melody_rnn_sequence_generator.get_generator_map()
    melody_rnn = generator_map[MODEL_NAME](checkpoint=None, bundle=bundle)
    melody_rnn.initialize()

    # 시드 설정 (Primer Melody)
    seed_sequence = music_pb2.NoteSequence()
    for i, pitch in enumerate(PRIMER_MELODY):
        seed_sequence.notes.add(
            pitch=pitch,
            start_time=i * 0.5,
            end_time=(i + 1) * 0.5,
            velocity=80
        )
    seed_sequence.total_time = len(PRIMER_MELODY) * 0.5

    # 5. 음악 생성 옵션 설정
    generator_options = generator_pb2.GeneratorOptions()
    generate_section = generator_options.generate_sections.add(
        start_time=seed_sequence.total_time,
        end_time=seed_sequence.total_time + (NUM_STEPS / 4.0) # 4분음표 기준
    )
    print(f"Generating {NUM_OUTPUTS} melodies...")
    
    # 날짜 및 시간 문자열 생성 (예: 250511_2030)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # MID로 저장
    for i in range(NUM_OUTPUTS):
        generated_sequence = melody_rnn.generate(seed_sequence, generator_options)
        output_file = os.path.join(OUTPUT_DIR, f'basic_rnn_gen_{timestamp}_{i+1}.mid')
        note_seq.sequence_proto_to_midi_file(generated_sequence, output_file)
        print(f"Saved: {output_file}")

    print("✅ MID 파일 생성 완료")

if __name__ == '__main__':
    main()
