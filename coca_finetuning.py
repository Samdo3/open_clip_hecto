import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder # ImageFolder를 활용할 수 있는 경우

# --- 설정 ---
dataset_base_dir = './hecto_data' # 사용자의 데이터셋 기본 경로
train_data_dir = os.path.join(dataset_base_dir, 'train')
test_data_dir = os.path.join(dataset_base_dir, 'test')
test_csv_path = os.path.join(dataset_base_dir, 'test.csv') # test.csv 파일 경로

# 이미지 크기 및 배치 크기 (나중에 CoCa 모델의 요구사항에 맞춰 조정 가능)
# 예시: open_clip의 coca_ViT-L-14는 224x224를 사용할 수 있음 (해당 JSON 파일 내 image_size 참고)
# CoCa 논문에서는 288x288을 사전 학습에 사용하고, 576x576으로 추가 학습하기도 함 (페이지 5, 7)
# 여기서는 일반적인 224로 시작하고, 추후 모델에 맞춰 조정
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0 # 환경에 맞게 조절

# --- 1. 최종 클래스 목록 확정 ---
# 클래스 통합 스크립트 실행 후의 train 폴더를 기준으로 클래스 목록 생성
if not os.path.exists(train_data_dir):
    print(f"오류: 학습 데이터 폴더를 찾을 수 없습니다: {train_data_dir}")
    # exit() # 실제 실행 시에는 여기서 중단
    # 예시를 위해 임시 클래스 생성 (실제로는 폴더 스캔)
    final_class_names = sorted([f'class_{i}' for i in range(5)])
else:
    final_class_names = sorted(os.listdir(train_data_dir))
    final_class_names = [d for d in final_class_names if os.path.isdir(os.path.join(train_data_dir, d))]

num_classes = len(final_class_names)
class_to_idx = {class_name: idx for idx, class_name in enumerate(final_class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

print(f"최종 클래스 개수: {num_classes}")
print(f"첫 5개 클래스: {final_class_names[:5]}")
print(f"클래스-인덱스 매핑 (일부): {list(class_to_idx.items())[:5]}")

# --- 2. 학습/검증 데이터 이미지 경로 및 레이블 수집 ---
all_image_paths = []
all_image_labels = []

if os.path.exists(train_data_dir):
    for class_name in final_class_names:
        class_dir = os.path.join(train_data_dir, class_name)
        image_files = glob.glob(os.path.join(class_dir, '*.jpg')) # 또는 다른 이미지 확장자
        for img_path in image_files:
            all_image_paths.append(img_path)
            all_image_labels.append(class_to_idx[class_name])
else:
    print(f"경고: {train_data_dir}가 존재하지 않아 임시 데이터로 진행합니다.")
    # 임시 데이터 생성 (실제로는 필요 없음)
    for i in range(num_classes * 10):
        all_image_paths.append(f"dummy_path_{i}.jpg")
        all_image_labels.append(i % num_classes)


print(f"전체 학습 이미지 수: {len(all_image_paths)}")

# --- 3. 학습/검증 데이터 분할 (층화 추출) ---
# 이미지 경로가 있어야 분할 가능
if all_image_paths:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths,
        all_image_labels,
        test_size=0.2, # 20%를 검증 세트로 사용
        random_state=42,
        stratify=all_image_labels # 레이블 비율을 유지하며 분할
    )
    print(f"학습 이미지 수: {len(train_paths)}, 검증 이미지 수: {len(val_paths)}")
else:
    print("분할할 학습 이미지가 없습니다.")
    train_paths, val_paths, train_labels, val_labels = [], [], [], []


# --- 4. 이미지 변환(Transforms) 정의 ---
# open_clip의 transform.py를 참고하여 mean/std 설정 가능
# 일반적으로 ImageNet 통계 사용 또는 모델별 사전 학습 시 사용된 값 사용
# 여기서는 일반적인 ImageNet 통계 사용
normalize_mean = (0.485, 0.456, 0.406)
normalize_std = (0.229, 0.224, 0.225)

# open_clip의 image_transform 함수를 보면 is_train에 따라 다르게 구성됨
# 일반적으로 사용하는 augmentation을 포함하여 구성
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

val_test_transform = transforms.Compose([
    transforms.Resize(256), # 일반적인 관례, IMG_SIZE보다 약간 크게
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])


# --- 5. PyTorch Dataset 클래스 정의 ---
class CarDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        if self.is_test: # 테스트 데이터의 경우 ID 추출을 위해 test.csv 사용
            self.test_df = pd.read_csv(test_csv_path)
            # img_path를 기준으로 ID를 매핑할 수 있도록 준비 (경로 형식이 다를 수 있으므로 basename 사용)
            self.path_to_id = {os.path.basename(row['img_path']): row['ID'] for _, row in self.test_df.iterrows()}


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # 임시로 검은색 이미지 반환 또는 에러 처리
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))


        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # img_path의 파일명 (예: TEST_00000.jpg)으로 ID 조회
            img_filename = os.path.basename(img_path)
            img_id = self.path_to_id.get(img_filename, None)
            if img_id is None:
                print(f"경고: 테스트 CSV에 없는 이미지 파일명입니다: {img_filename}")
            return image, img_id
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)

# --- 6. 데이터 로더(DataLoaders) 생성 ---

# 학습 및 검증 데이터셋 생성
if train_paths and val_paths:
    train_dataset = CarDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CarDataset(val_paths, val_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"학습 데이터로더 준비 완료. 배치 크기: {BATCH_SIZE}")
    print(f"검증 데이터로더 준비 완료. 배치 크기: {BATCH_SIZE}")

    # 데이터로더 샘플 확인 (선택 사항)
    # images, labels = next(iter(train_loader))
    # print(f"학습 데이터 샘플 이미지 배치 shape: {images.shape}")
    # print(f"학습 데이터 샘플 레이블 배치 shape: {labels.shape}")
else:
    print("학습 또는 검증 데이터가 없어 데이터로더를 생성할 수 없습니다.")
    train_loader, val_loader = None, None

# 평가(Test) 데이터셋 및 데이터로더 생성
if os.path.exists(test_csv_path) and os.path.exists(test_data_dir):
    test_info_df = pd.read_csv(test_csv_path)
    # test.csv의 img_path는 상대 경로일 수 있으므로, test_data_dir 기준으로 절대 경로 생성
    # 예시: 'test_data_dir/TEST_00000.jpg'
    test_image_paths = [os.path.join(test_data_dir, os.path.basename(p)) for p in test_info_df['img_path']]

    # 실제 파일 존재 여부 확인 (선택적이지만 권장)
    test_image_paths_exist = [p for p in test_image_paths if os.path.exists(p)]
    if len(test_image_paths_exist) != len(test_image_paths):
        print(f"경고: test.csv에 명시된 이미지 중 일부를 찾을 수 없습니다. (존재: {len(test_image_paths_exist)}, 명시: {len(test_image_paths)})")


    test_dataset = CarDataset(test_image_paths_exist, transform=val_test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"테스트 데이터로더 준비 완료. 이미지 수: {len(test_dataset)}, 배치 크기: {BATCH_SIZE}")

    # 테스트 데이터로더 샘플 확인 (선택 사항)
    # images, ids = next(iter(test_loader))
    # print(f"테스트 데이터 샘플 이미지 배치 shape: {images.shape}")
    # print(f"테스트 데이터 샘플 ID 배치: {ids}")

else:
    print("테스트 CSV 파일 또는 테스트 이미지 폴더를 찾을 수 없어 테스트 데이터로더를 생성할 수 없습니다.")
    test_loader = None

# --- 추가: 원래 클래스명과 통합된 클래스명 매핑 정보 (제출 시 필요) ---
# 사용자가 제공한 class_map 정보를 바탕으로 생성
# 예시 (실제로는 사용자의 class_map_X 변수를 활용해야 함):
original_to_representative_map = {
    "K5_하이브리드_3세대_2020_2023": "K5_3세대_하이브리드_2020_2022",
    "디_올_뉴_니로_2022_2025": "디_올뉴니로_2022_2025",
    "박스터_718_2017_2024": "718_박스터_2017_2024",
    "라브4_4세대_2013_2018": "RAV4_2016_2018",
    "라브4_5세대_2019_2024": "RAV4_5세대_2019_2024"
}

# 제출 시 필요한 모든 원본 클래스 목록 (sample_submission.csv의 컬럼 순서대로)
# 이 부분은 sample_submission.csv를 직접 읽어서 컬럼명을 가져오는 것이 가장 정확합니다.
# 예시:
# submission_df_template = pd.read_csv('path_to_sample_submission.csv')
# all_original_submission_columns = submission_df_template.columns[1:].tolist() # ID 제외

# 이 스크립트에서는 일단 final_class_names (통합된 클래스)와 original_to_representative_map을
# 활용하여 추후 제출 파일 생성 단계에서 사용합니다.

print("\n1단계: 데이터 준비 및 전처리 완료.")


# --- 2단계: 모델 준비 및 설정 ---
import torch
import torch.nn as nn
import open_clip


# 1. 사전 학습된 CoCa 모델 로드
coca_model_name="coca_ViT-L-14"
pretrained_dataset = "laion2b_s13b_b90k" # PRETRAINED.md에서 CoCa 모델의 가중치 이름 확인

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print(f"Loading CoCa model: {coca_model_name} with pretrained weights: {pretrained_dataset}")
    # open_clip.create_model_and_transforms 함수는 모델과 이미지 전처리 함수를 반환합니다.
    # 이미지 전처리는 1단계의 transforms 정의 시 참고했거나, 여기서 반환된 preprocess를 사용할 수 있습니다.
    # 여기서는 모델만 로드하고, 이미지 전처리는 1단계에서 정의한 것을 사용한다고 가정합니다.
    # 만약 1단계의 transforms 대신 여기서 반환된 preprocess를 사용하려면 해당 부분을 수정해야 합니다.
    coca_model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name=coca_model_name,
        pretrained=pretrained_dataset,
        device=device
    )

    print(f"{coca_model_name} loaded successfully.")

    # (선택 사항) 모델의 일부 레이어만 학습하도록 설정 (예: 이미지 인코더의 일부만 fine-tuning)
    # for name, param in coca_model.named_parameters():
    #     if "visual" not in name: # 시각적 백본(visual backbone)이 아닌 부분은 고정
    #         param.requires_grad = False
    #     elif "visual.ln_post" in name or "visual.proj" in name : # 예시: 마지막 레이어들만 학습
    #          param.requires_grad = True
    #     elif "visual.transformer.resblocks.23" in name: # 예시: ViT-L-14의 마지막 블록만 학습
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False # 그 외 시각적 백본 레이어는 고정

except Exception as e:
    print(f"Error loading CoCa model: {e}")
    print("Please ensure the model name and pretrained dataset tag are correct and network is available if downloading.")
    coca_model = None


# 2. 분류기(Classifier) 추가
if coca_model is not None:
    # CoCa 모델의 이미지 인코더 출력 차원 확인
    # ViT-L-14의 경우 이미지 특징 차원은 768일 수 있습니다. (CoCa 논문 Table 1 또는 open_clip 코드 확인)
    # 임시 더미 이미지로 특징 차원 확인

    try:
        dummy_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad():
            # CoCa 모델은 이미지를 직접 입력받아 이미지 임베딩을 생성합니다.
            # CLIP과 유사하게 image_encoder를 직접 호출하거나, CoCa 클래스의 encode_image 사용
            # open_clip의 CoCa 구현에서는 model.encode_image(image)를 사용합니다.
            image_features = coca_model.encode_image(dummy_image) # (batch_size, embedding_dim)
        
        # image_features는 풀링된 [CLS] 토큰 또는 평균 풀링된 특징일 수 있습니다.
        # CoCa의 경우, `image_features`는 attentional pooler를 거친 결과일 것입니다.
        # CoCa.pdf "Attentional Poolers" 섹션 참고
        # "contrastive loss n_query=1"
        
        image_feature_dim = image_features.shape[-1]
        print(f"Detected image feature dimension: {image_feature_dim}")

        # 분류기 정의
        # 간단한 선형 레이어 또는 여러 레이어로 구성 가능
        classifier = nn.Linear(image_feature_dim, num_classes)
        classifier = classifier.to(device)

        print(f"Classifier added: Linear({image_feature_dim}, {num_classes})")

        # (선택 사항) 모델 전체를 하나의 nn.Sequential로 묶거나, forward 메소드에서 처리
        # 여기서는 CoCa 모델과 분류기를 별도로 두고, 학습 루프에서 연결하여 사용합니다.

    except Exception as e:
        print(f"Error detecting feature dimension or creating classifier: {e}")
        classifier = None
else:
    classifier = None

# 예시: 모델 파라미터 확인 (학습 가능한 파라미터 수)
if coca_model and classifier:
    total_params = sum(p.numel() for p in coca_model.parameters())
    trainable_params = sum(p.numel() for p in coca_model.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    print(f"\nTotal CoCa Model parameters: {total_params:,}")
    print(f"Trainable CoCa Model parameters: {trainable_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total trainable parameters (CoCa + Classifier): {trainable_params + classifier_params:,}")
    print("2-1단계 : 모델 및 분류기 준비 완료.")

# --- 3단계: 모델 학습 (파인튜닝) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # 학습률 스케줄러 예시
import time
import copy # 모델 가중치 복사용
from sklearn.metrics import log_loss # 평가 지표
import numpy as np # log_loss 계산 시 필요
from tqdm import tqdm # tqdm 임포트
# 1. 손실 함수(Loss Function) 정의
# 다중 클래스 분류이므로 CrossEntropyLoss 사용
criterion = nn.CrossEntropyLoss()
print(f"손실 함수: CrossEntropyLoss")

# 2. 옵티마이저(Optimizer) 선택
# CoCa 모델의 파라미터와 새로 추가한 분류기의 파라미터를 함께 최적화

# (선택 사항) 파라미터 그룹별로 다른 학습률 설정 가능
# params_to_optimize = [
#     {"params": coca_model.parameters(), "lr": 1e-5}, # 사전 학습된 부분은 작은 학습률
#     {"params": classifier.parameters(), "lr": 1e-4}  # 새로 추가된 분류기는 상대적으로 큰 학습률
# ]
# optimizer = optim.AdamW(params_to_optimize, weight_decay=0.01)

# 모든 학습 가능한 파라미터를 하나의 옵티마이저로 관리
# coca_model의 requires_grad가 True인 파라미터와 classifier의 파라미터를 합쳐서 전달
trainable_coca_params = [p for p in coca_model.parameters() if p.requires_grad]
all_trainable_params = trainable_coca_params + list(classifier.parameters())

learning_rate = 1e-5 # 시작 학습률 (조정 필요)
optimizer = optim.AdamW(all_trainable_params, lr=learning_rate, weight_decay=0.01)
print(f"옵티마이저: AdamW, 학습률: {learning_rate}")

# 3. 학습 스케줄러(Learning Rate Scheduler) 설정 (선택 사항)
# 예시: 검증 손실이 개선되지 않으면 학습률을 감소시키는 ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
print(f"학습률 스케줄러: ReduceLROnPlateau (patience=3, factor=0.1)")

# 4. 학습 루프(Training Loop) 구현
num_epochs = 20  # 총 학습 에폭 수 (조정 필요)
best_model_wts_coca = None
best_model_wts_classifier = None
best_val_logloss = float('inf') # 가장 낮은 검증 LogLoss를 찾기 위함
log_loss_calculation_failed = False # LogLoss 계산 실패 여부 플래그

# 학습 및 검증 결과를 저장할 리스트
train_loss_history = []
val_loss_history = []
val_logloss_history = [] # 대회 평가 지표인 LogLoss 기록

print(f"\n--- 모델 학습 시작 (총 에폭: {num_epochs}) ---")

for epoch in range(num_epochs):
    epoch_start_time=time.time()
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print('-' * 10)

    # 각 에폭은 학습 단계와 검증 단계를 가짐
    for phase in ['train', 'val']:
        if phase == 'train':
            coca_model.train()  # 모델을 학습 모드로 설정
            classifier.train()  # 분류기도 학습 모드로 설정
            dataloader = train_loader
            progress_bar_desc = "Training"
        else:
            coca_model.eval()   # 모델을 평가 모드로 설정
            classifier.eval()   # 분류기도 평가 모드로 설정
            dataloader = val_loader
            progress_bar_desc = "Validation"

        running_loss = 0.0
        # LogLoss 계산을 위한 예측 확률과 실제 레이블 저장 리스트 (검증 단계용)
        all_preds_probs_val = []
        all_labels_val = []

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{progress_bar_desc} Epoch {epoch+1}/{num_epochs}", unit="batch")


        # 데이터 로더에서 미니배치 반복
        for batch_idx, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 옵티마이저 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파 (forward)
            # 학습 단계에서만 그래디언트 계산 활성화
            with torch.set_grad_enabled(phase == 'train'):
                # CoCa 모델로 이미지 특징 추출
                # open_clip의 CoCa는 model.encode_image(image) 형태로 이미지 특징 반환
                image_features = coca_model.encode_image(inputs) # (batch_size, image_feature_dim)

                # 분류기로 클래스 예측
                outputs = classifier(image_features) # (batch_size, num_merged_classes)
                
                # 손실 계산 (CrossEntropyLoss는 내부적으로 Softmax를 포함)
                loss = criterion(outputs, labels)

                # 학습 단계인 경우 역전파 + 옵티마이저 실행
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)

            # 검증 단계에서 LogLoss 계산을 위한 값 저장
            if phase == 'val':
                # Softmax를 적용하여 확률값 얻기
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                all_preds_probs_val.extend(probs)
                all_labels_val.extend(labels.cpu().numpy())
            
            # tqdm의 postfix를 사용하여 현재 손실 표시
            progress_bar.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{running_loss / ((batch_idx + 1) * dataloader.batch_size):.4f}')



        epoch_loss = running_loss / len(dataloader.dataset)

        if phase == 'train':
            train_loss_history.append(epoch_loss)
            print(f"\n{progress_bar_desc} - Train CE Loss: {epoch_loss:.4f}") # tqdm 줄바꿈을 위해 \n 추가
        else: # phase == 'val'
            val_loss_history.append(epoch_loss)
            print(f"\n{progress_bar_desc} - Validation CE Loss: {epoch_loss:.4f}") # tqdm 줄바꿈을 위해 \n 추가

            # 검증 데이터에 대한 LogLoss 계산 (대회 평가지표)
            log_loss_calculation_failed = False # 매 에폭마다 초기화
            try:
                # all_preds_probs_val는 (num_samples, num_classes) 형태의 numpy 배열
                # all_labels_val는 (num_samples,) 형태의 numpy 배열 (정수 레이블)
                
                # 확률 정규화 및 클리핑 (제출 코드의 로직과 유사하게)
                y_pred_probs_val = np.array(all_preds_probs_val)
                # y_pred_probs_val = y_pred_probs_val / np.sum(y_pred_probs_val, axis=1, keepdims=True) # DataLoader에서 나온 확률은 이미 합이 1일 것임. 확인 필요.
                y_pred_probs_val_clipped = np.clip(y_pred_probs_val, 1e-15, 1 - 1e-15)
                
                # 실제 레이블(0부터 num_classes-1까지)과 예측 확률
                # labels 인자는 log_loss 함수에 클래스의 범위를 알려줌
                val_logloss = log_loss(all_labels_val, y_pred_probs_val_clipped, labels=list(range(num_classes)))
                val_logloss_history.append(val_logloss)
                print(f"Validation LogLoss: {val_logloss:.4f}")

                # 학습률 스케줄러 업데이트 (검증 LogLoss 기준)
                scheduler.step(val_logloss)

                # 가장 좋은 모델 저장 (LogLoss 기준)
                if val_logloss < best_val_logloss:
                    best_val_logloss = val_logloss
                    # 모델의 state_dict를 깊은 복사하여 저장
                    best_model_wts_coca = copy.deepcopy(coca_model.state_dict())
                    best_model_wts_classifier = copy.deepcopy(classifier.state_dict())
                    torch.save({
                        'epoch': epoch,
                        'coca_model_state_dict': best_model_wts_coca,
                        'classifier_state_dict': best_model_wts_classifier,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_logloss': best_val_logloss,
                        'class_to_idx': class_to_idx, # 클래스 매핑 정보도 함께 저장
                        'merged_class_names': final_class_names
                    }, f'best_coca_finetuned_model_epoch{epoch+1}_logloss{best_val_logloss:.4f}.pth')
                    print(f"Saved Best Model (LogLoss: {best_val_logloss:.4f}) at Epoch {epoch+1}")

            except ValueError as e:
                print(f"LogLoss 계산 중 오류 발생: {e}")
                print("예측 확률값의 합이 1이 아니거나, NaN/inf가 포함되었을 수 있습니다.")
                log_loss_calculation_failed = True # LogLoss 계산 실패 플래그 설정
                # 이 경우 scheduler.step() 호출 및 모델 저장 로직을 건너뛸 수 있음

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} 완료. 소요 시간: {epoch_duration // 60:.0f}m {epoch_duration % 60:.0f}s")
    if log_loss_calculation_failed:
        print("경고: 이번 에폭에서 LogLoss 계산에 실패했습니다. 학습률 스케줄러 업데이트 및 최적 모델 저장이 수행되지 않았을 수 있습니다.")

print(f"\n--- 모델 학습 완료 ---")
if best_val_logloss != float('inf'):
    print(f"가장 낮은 검증 LogLoss: {best_val_logloss:.4f}")
else:
    print("학습 중 유효한 LogLoss를 기록하지 못했습니다.")

# 가장 좋은 모델 가중치 로드 (만약 필요하다면)
# if best_model_wts_coca and best_model_wts_classifier:
#     coca_model.load_state_dict(best_model_wts_coca)
#     classifier.load_state_dict(best_model_wts_classifier)
#     print("Best model weights loaded.")

# --- 학습 결과 시각화 (선택 사항, matplotlib 필요) ---
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "c:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
# 음수 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train CE Loss')
plt.plot(val_loss_history, label='Validation CE Loss')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss')
plt.legend()
plt.title('Training and Validation CE Loss')

plt.subplot(1, 2, 2)
plt.plot(val_logloss_history, label='Validation LogLoss', color='red')
plt.xlabel('Epoch')
plt.ylabel('LogLoss')
plt.legend()
plt.title('Validation LogLoss')
plt.tight_layout()
plt.show()
plt.savefig('training_curves.png')