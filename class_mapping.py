import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import open_clip

# --- 1. 기본 설정 ---
dataset_base_dir = './' # 현재 스크립트 위치를 기준으로 가정
# 제로샷 평가를 수행할 이미지들이 있는 폴더 (통합된 학습 데이터 폴더 또는 별도의 검증/테스트 폴더)
# 여기서는 통합된 학습 데이터 폴더를 사용한다고 가정 (클래스명 추출 및 이미지 로드 목적)
image_folder_for_eval_name = 'hecto_data/train' # 또는 'validation_set' 등
image_data_path_for_eval = os.path.join(dataset_base_dir, image_folder_for_eval_name)

model_name_to_use = "ViT-H-14-378-quickgelu"
pretrained_tag = "dfn5b"

try:
    model, _, image_preprocess = open_clip.create_model_and_transforms(
        model_name=model_name_to_use,
        pretrained=pretrained_tag # 모델 가중치를 불러오기 위한 태그
    )
    tokenizer = open_clip.get_tokenizer(model_name_to_use) # 제로샷 분류기 생성 시 필요
    print(f"'{model_name_to_use}' 모델 (pretrained: '{pretrained_tag}') 및 전처리 함수, 토크나이저를 성공적으로 로드했습니다.")
except Exception as e:
    print(f"OpenCLIP 모델 로드 중 오류 발생: {e}")
    print("계속 진행할 수 없습니다. 모델명과 pretrained 태그를 확인하세요.")
    exit()

# 제로샷 평가 시에는 데이터 증강을 사용하지 않습니다.
eval_transforms = image_preprocess

# --- 2. ImageFolder를 사용하여 평가할 데이터셋 및 클래스 정보 가져오기 ---
if not os.path.exists(image_data_path_for_eval):
    print(f"오류: 평가용 이미지 데이터 폴더 '{image_data_path_for_eval}'를 찾을 수 없습니다.")
    exit()

try:
    eval_dataset = datasets.ImageFolder(
        root=image_data_path_for_eval,
        transform=eval_transforms
    )
    print(f"평가용 데이터셋 로드 완료. 총 이미지 수: {len(eval_dataset)}")
    print(f"평가용 데이터셋 클래스 수: {len(eval_dataset.classes)}")
except Exception as e:
    print(f"평가용 데이터셋 ImageFolder 생성 중 오류 발생: {e}")
    exit()

# --- 3. 클래스명과 인덱스 매핑 사전 생성 및 저장 (제로샷 분류기 생성 시 필요) ---
class_to_idx = eval_dataset.class_to_idx
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

mapping_save_path = os.path.join(dataset_base_dir, f'class_mappings_for_eval.pkl')
try:
    with open(mapping_save_path, 'wb') as f:
        pickle.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'classes': eval_dataset.classes # 정렬된 클래스명 리스트도 함께 저장
        }, f)
    print(f"평가용 클래스 매핑 사전 저장 완료: {mapping_save_path}")
    print("클래스명 리스트 (eval_dataset.classes):", eval_dataset.classes)
    # print("class_to_idx:", class_to_idx) # 필요시 주석 해제
except Exception as e:
    print(f"평가용 클래스 매핑 사전 저장 중 오류 발생: {e}")

# --- 4. DataLoader 생성 (제로샷 평가 실행 시 사용) ---
batch_size = 32  # 제로샷 평가는 일반적으로 더 큰 배치 크기 사용 가능
num_workers = 0

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False, # 평가는 순서대로
    num_workers=num_workers,
    pin_memory=True
)

print(f"Eval DataLoader 생성 완료. 1배치 크기: {batch_size}")
