import sys
import os
import torch
import pickle
import open_clip # OpenCLIP 라이브러리
from open_clip_train.precision import get_autocast # 사용자의 임포트 유지
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset # TestCarDataset을 위해 Dataset 추가
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm
from PIL import Image # TestCarDataset에서 이미지 로드를 위해
import contextlib 


# 사용자의 경로 설정 유지
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = current_script_dir
sys.path.insert(0, project_root_dir)


# --- 1. 기본 설정 ---
dataset_base_dir = './hecto_data'
test_image_folder_name = 'test'
test_csv_path = os.path.join(dataset_base_dir, 'test.csv')
sample_submission_path = os.path.join(dataset_base_dir, 'sample_submission.csv')
# 대표 클래스명과 인덱스 매핑 정보 (Phase 1에서 저장한 파일명과 일치해야 함)
# 이 파일은 제로샷 분류기 생성 시 car_classnames를 가져오는 데 사용됩니다.
class_mapping_file_name = 'class_mappings_for_eval.pkl'
class_mapping_save_path = os.path.join('./', class_mapping_file_name)

model_name_to_use = "ViT-H-14-378-quickgelu"
pretrained_tag = "dfn5b" # 이 태그로 사전 훈련된 모델 가중치를 바로 로드

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
args_precision = 'amp'
args_device = device

# --- 2. 모델 로드 (사전 훈련된 가중치 직접 로드) ---
try:
    model, _, image_preprocess = open_clip.create_model_and_transforms(
        model_name=model_name_to_use,
        pretrained=pretrained_tag # 사전 훈련된 가중치를 바로 로드
    )
    tokenizer = open_clip.get_tokenizer(model_name_to_use)
    print(f"'{model_name_to_use}' 모델 (pretrained: '{pretrained_tag}')을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"OpenCLIP 모델 로드 중 오류 발생: {e}")
    exit()

model.to(device)
model.eval() # 추론 모드로 설정

# --- 3. 클래스 매핑 정보 로드 (제로샷 분류기 생성용) ---
if not os.path.exists(class_mapping_save_path):
    print(f"오류: 클래스 매핑 파일 '{class_mapping_save_path}'를 찾을 수 없습니다.")
    exit()
try:
    with open(class_mapping_save_path, 'rb') as f:
        mappings = pickle.load(f)
    car_classnames = mappings['classes'] # ImageFolder에서 추출한 대표 클래스명 리스트
    idx_to_representative_class = mappings['idx_to_class'] # 추론 결과 매핑에 사용
    num_representative_classes = len(car_classnames)
    print(f"클래스 매핑 정보 로드 완료. 대표 클래스 수: {num_representative_classes}")
except Exception as e:
    print(f"클래스 매핑 파일 로드 중 오류 발생: {e}")
    exit()

# --- 4. 동일 클래스 매핑 정의 (대표 클래스 -> 원본 제출용 클래스 리스트) ---
representative_to_original_map = {
    "K5_3세대_하이브리드_2020_2022": ['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'],
    "디_올뉴니로_2022_2025": ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'],
    "718_박스터_2017_2024": ['718_박스터_2017_2024', '박스터_718_2017_2024'],
    "RAV4_2016_2018": ['RAV4_2016_2018', '라브4_4세대_2013_2018'],
    "RAV4_5세대_2019_2024": ['RAV4_5세대_2019_2024', '라브4_5세대_2019_2024']
}
for rep_class_name in car_classnames: # 학습 시 사용된 모든 대표 클래스에 대해
    if rep_class_name not in representative_to_original_map:
        representative_to_original_map[rep_class_name] = [rep_class_name]

# --- 5. 제로샷 분류를 위한 텍스트 템플릿 정의 ---
car_classification_templates = (
    lambda c: f'a photo of a {c}.',
    lambda c: f'a picture of a {c}.',
    lambda c: f'an image of a {c}.',
    lambda c: f'a photograph of a {c}.',
    # 'car', 'vehicle', 'automobile', 'model' 명시
    lambda c: f'a photo of the car: {c}.',
    lambda c: f'a picture of the vehicle: {c}.',
    lambda c: f'an image of the automobile: {c}.',
    lambda c: f'a photograph of the car model: {c}.',
    lambda c: f'a photo of a {c} car.',
    lambda c: f'a picture of a {c} vehicle.',
    lambda c: f'an image of a {c} automobile.',
    lambda c: f'this is a {c}.',
    lambda c: f'this car is a {c}.',
    lambda c: f'this vehicle is a {c}.',
    lambda c: f'this automobile is a {c}.',
    lambda c: f'the car model is {c}.',
    lambda c: f'vehicle type: {c}.',
    lambda c: f'automobile model: {c}.',
    # 약간 더 구체적인 문맥 (하지만 여전히 일반적)
    lambda c: f'a clear photo of a {c}.',
    lambda c: f'a high-resolution image of a {c}.',
    lambda c: f'a standard photo of a {c} car.',
    lambda c: f'a front view of a {c}.', # 각도에 대한 일반적 언급 (특정 각도 강제 X)
    lambda c: f'a side view of a {c}.',  # "
    lambda c: f'a rear view of a {c}.',   # "
    lambda c: f'a photo displaying a {c}.',
    lambda c: f'an image showcasing a {c}.',
    # OpenAI 템플릿에서 차용하되, 현실 사진에 맞게 일부 수정/선별
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a close-up photo of a {c}.', # 너무 과한 클로즈업이 아니라면 사용 가능
    lambda c: f'a photo of the {c}.', # 'the' 사용
    # 중립적이거나 약간 긍정적인 품질 묘사
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a clean {c}.', # 차량이 깨끗하다는 가정
    # 크기에 대한 언급 (일반적인 크기를 상상하도록 유도)
    lambda c: f'a photo of a standard size {c}.',
)

# --- 6. 제로샷 분류기 가중치 생성 ---
print("제로샷 분류기 생성을 시작합니다...")
try:
    with torch.inference_mode():
        zeroshot_weights = open_clip.build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=car_classnames,
            templates=car_classification_templates,
            device=device,
            num_classes_per_batch=10,
            use_tqdm=True
        )
    print("제로샷 분류기 가중치 생성 완료.")
    print("생성된 가중치 텐서의 형태:", zeroshot_weights.shape)
except Exception as e:
    print(f"제로샷 분류기 생성 중 오류 발생: {e}")
    exit()

# --- 7. 테스트 데이터셋 및 DataLoader 정의 ---
class TestCarDataset(Dataset):
    def __init__(self, csv_file, root_dir_base, transform=None): # root_dir_base로 변경
        self.annotations_df = pd.read_csv(csv_file)
        self.root_dir_base = root_dir_base # dataset_base_dir (예: './')
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        img_path_relative_to_base = self.annotations_df.iloc[idx]['img_path'] # 예: "hecto_data/test/TEST_00000.jpg"
        img_full_path = os.path.join(self.root_dir_base, img_path_relative_to_base)

        try:
            image = Image.open(img_full_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"경고: 테스트 이미지 파일을 찾을 수 없습니다 - {img_full_path}")
            # 견고한 처리를 위해, 여기서 None을 반환하고 DataLoader의 collate_fn에서 처리하거나,
            # 또는 더미 이미지를 반환하고 ID와 함께 해당 ID는 나중에 제외하는 등의 처리가 필요할 수 있습니다.
            # 간단하게는 에러를 발생시키거나, 빈 텐서를 반환합니다.
            # 여기서는 첫 번째 학습 이미지로 대체하는 대신, None을 반환하도록 수정하고 추후 처리합니다.
            # 또는 프로그램 중단: raise FileNotFoundError(f"이미지 없음: {img_full_path}")
            # 이 예제에서는 간단히 첫번째 학습 이미지로 대체하는 대신, 빈 텐서를 만듭니다. (실제로는 더 나은 처리 필요)
            return torch.zeros((3, image_preprocess.transforms[0].size[0], image_preprocess.transforms[0].size[1])), self.annotations_df.iloc[idx]['ID']


        img_id = self.annotations_df.iloc[idx]['ID']
        return image, img_id

if not os.path.exists(test_csv_path):
    print(f"오류: 테스트 CSV 파일 '{test_csv_path}'를 찾을 수 없습니다.")
    exit()

# --- 평가 실행 함수 정의 ---
def calculate_accuracy_logloss(model, classifier_weights, dataloader, device, precision_str, num_classes):
    """
    주어진 모델, 분류기 가중치, 데이터로더를 사용하여 정확도와 LogLoss를 계산합니다.
    이 함수는 정답 레이블(targets)이 있는 데이터로더(예: 학습 데이터나 검증 데이터로 만든 로더)에 사용됩니다.
    """
    current_autocast_context_manager = None
    current_input_dtype = None

    # OPENCLIP_PRECISION_UTILS_AVAILABLE 플래그 대신 직접 get_autocast 사용 시도
    try:
        # from open_clip_train.precision import get_autocast # 스크립트 상단에서 이미 임포트 가정
        current_autocast_context_manager = get_autocast(precision_str, device_type=torch.device(device).type)
        current_input_dtype = open_clip.get_input_dtype(precision_str) if hasattr(open_clip, 'get_input_dtype') else torch.float32
        if not hasattr(open_clip, 'get_input_dtype'):
            print("Warning: open_clip.get_input_dtype을 찾을 수 없어 torch.float32를 기본 입력 dtype으로 사용합니다.")
            if precision_str == "amp" or precision_str == "fp16": current_input_dtype = torch.float16
            elif precision_str == "bf16": current_input_dtype = torch.bfloat16


    except NameError: # get_autocast가 정의되지 않은 경우 (임포트 실패 등)
        print("Warning: get_autocast를 찾을 수 없습니다 (NameError). PyTorch 기본 AMP 기능을 사용합니다.")
        if device == "cuda" and (precision_str == "amp" or precision_str == "fp16"):
            current_autocast_context_manager = torch.cuda.amp.autocast()
            current_input_dtype = torch.float16
        elif device == "cuda" and precision_str == "bf16":
            current_autocast_context_manager = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            current_input_dtype = torch.bfloat16
        else: # fp32 or cpu
            @contextlib.contextmanager
            def autocast_nop_local():
                yield
            current_autocast_context_manager = autocast_nop_local()
            current_input_dtype = torch.float32
    
    model.eval()
    all_logits_list = []
    all_targets_list = [] # 정답 레이블을 수집
    
    try:
        logit_scale = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 100.0
    except:
        logit_scale = 100.0
    print(f"Using logit_scale (for eval): {logit_scale}")

    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Calculating LogLoss/Accuracy"): # dataloader는 target을 포함해야 함
            images = images.to(device=device, dtype=current_input_dtype if device != 'cpu' else torch.float32)
            targets = targets.to(device) # 정답 레이블도 디바이스로

            with current_autocast_context_manager(): # 수정: 괄호 없이 사용
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = logit_scale * image_features @ classifier_weights

            all_logits_list.append(logits.cpu())
            all_targets_list.append(targets.cpu()) # 정답 레이블 수집

    if not all_logits_list:
        print("평가 데이터가 없습니다 (for LogLoss/Accuracy).")
        return 0.0, 0.0, float('inf')

    all_logits_tensor = torch.cat(all_logits_list)
    all_targets_tensor = torch.cat(all_targets_list)

    # 정확도 계산
    pred_top5 = all_logits_tensor.topk(5, 1, True, True)[1].t()
    correct_top5 = pred_top5.eq(all_targets_tensor.view(1, -1).expand_as(pred_top5))
    acc1 = float(correct_top5[:1].reshape(-1).float().sum(0, keepdim=True).numpy())
    acc5 = float(correct_top5[:5].reshape(-1).float().sum(0, keepdim=True).numpy())
    n_samples = all_targets_tensor.size(0)
    top1_accuracy = acc1 / n_samples
    top5_accuracy = acc5 / n_samples

    # LogLoss 계산 (대회 제공 로직 적용)
    probabilities_from_model = torch.softmax(all_logits_tensor, dim=1).numpy()
    prob_sums = probabilities_from_model.sum(axis=1, keepdims=True)
    probabilities_normalized = np.divide(probabilities_from_model, prob_sums, 
                                         out=np.full_like(probabilities_from_model, 1/num_classes), 
                                         where=prob_sums!=0)
    y_pred_for_logloss = np.clip(probabilities_normalized, 1e-15, 1 - 1e-15)
    true_labels_np = all_targets_tensor.numpy()
    
    try:
        logloss_score = log_loss(true_labels_np, y_pred_for_logloss, labels=list(range(num_classes)))
    except ValueError as e:
        print(f"LogLoss 계산 중 오류: {e}")
        logloss_score = float('inf')

    return top1_accuracy, top5_accuracy, logloss_score


# TestCarDataset의 root_dir_base는 test.csv 내의 img_path가 시작되는 기준 경로입니다.
# test.csv의 img_path가 "hecto_data/test/TEST_00000.jpg" 형태라면, root_dir_base는 './' (dataset_base_dir) 입니다.
test_dataset = TestCarDataset(
    csv_file=test_csv_path,
    root_dir_base=dataset_base_dir, # test.csv의 img_path가 이 경로를 기준으로 시작됨
    transform=image_preprocess
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
print(f"Test DataLoader 생성 완료. 총 테스트 이미지 수: {len(test_dataset)}")

# --- LogLoss 평가를 위한 준비 및 실행 ---
eval_data_for_logloss_path = os.path.join(dataset_base_dir, 'train') # 정답 레이블이 있는 폴더

if os.path.exists(eval_data_for_logloss_path):
    print(f"\nLogLoss 평가를 위해 '{eval_data_for_logloss_path}' 데이터를 사용합니다.")
    logloss_eval_dataset = datasets.ImageFolder(
        root=eval_data_for_logloss_path,
        transform=image_preprocess # 테스트/추론 시와 동일한 전처리
    )
    logloss_eval_loader = DataLoader(
        logloss_eval_dataset,
        batch_size=64, # 이전 eval_loader와 동일한 배치 크기 또는 적절히 조정
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if len(car_classnames) != len(logloss_eval_dataset.classes):
        print("경고: 제로샷 분류기와 LogLoss 평가 데이터셋의 클래스 수가 다릅니다!")
        print(f"제로샷 분류기 클래스 ({len(car_classnames)}): {car_classnames[:3]}...")
        print(f"LogLoss 평가 데이터셋 클래스 ({len(logloss_eval_dataset.classes)}): {logloss_eval_dataset.classes[:3]}...")

    print(f"\nLogLoss 및 정확도 평가를 시작합니다 (데이터: {eval_data_for_logloss_path})...")
    eval_top1_acc, eval_top5_acc, eval_logloss_val = calculate_accuracy_logloss(
        model,
        zeroshot_weights, # 위에서 생성한 제로샷 분류기 가중치
        logloss_eval_loader, # 정답 레이블이 있는 데이터 로더
        device,
        args_precision,
        num_representative_classes # 대표 클래스의 수
    )

    print(f"\n--- 제로샷 성능 평가 결과 (데이터: {os.path.basename(eval_data_for_logloss_path)}) ---")
    print(f"Top-1 Accuracy: {eval_top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {eval_top5_acc * 100:.2f}%")
    print(f"LogLoss (대회 방식 유사): {eval_logloss_val:.4f}")


# --- 8. 모델 추론 (제로샷 방식) 및 결과 저장 ---
autocast_context_manager = get_autocast(args_precision, device_type=torch.device(device).type)
current_input_dtype = open_clip.get_input_dtype(args_precision) if hasattr(open_clip, 'get_input_dtype') else torch.float32

all_image_ids = []
all_probabilities_representative = [] # 각 대표 클래스에 대한 확률 저장

try:
    logit_scale = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 100.0
except:
    logit_scale = 100.0
print(f"Using logit_scale for inference: {logit_scale}")

with torch.inference_mode():
    for images, image_ids in tqdm(test_loader, desc="Inferencing"):
        # TestCarDataset의 __getitem__에서 오류 발생 시 images가 더미 텐서일 수 있음 (실제로는 None 처리 또는 에러 핸들링)
        if images is None: # FileNotFoundError 등으로 이미지를 못 읽은 경우 스킵 (TestCarDataset 수정 필요)
            # all_image_ids.extend(list(image_ids)) # ID는 추가하되
            # all_probabilities_representative.append(np.full((len(image_ids), num_representative_classes), 1/num_representative_classes)) # 균등 확률로 채움
            print(f"경고: 일부 이미지 로드 실패, 해당 ID: {list(image_ids)}")
            # 이 경우, submission 파일에 해당 ID에 대한 처리를 어떻게 할지 결정해야 합니다. (예: 평균 확률, 0 등)
            # 현재는 정상적으로 로드되었다고 가정하고 진행합니다.
            pass


        images = images.to(device=device, dtype=current_input_dtype if device != 'cpu' else torch.float32)

        with autocast_context_manager():
            output = model(image=images)
            if isinstance(output, dict):
                image_features = output['image_features']
            else:
                image_features = output[0]
            
            # 제로샷 로짓 계산 (이미지 특징과 제로샷 텍스트 특징 내적)
            logits_representative = logit_scale * image_features @ zeroshot_weights

        probabilities_rep = torch.softmax(logits_representative, dim=1)
        all_image_ids.extend(list(image_ids)) # image_ids가 텐서일 경우 .tolist() 또는 list()
        all_probabilities_representative.append(probabilities_rep.cpu().numpy())

if not all_probabilities_representative:
    print("오류: 추론 결과가 없습니다.")
    exit()

all_probabilities_np = np.concatenate(all_probabilities_representative, axis=0)
print(f"추론 완료. 예측된 확률의 형태: {all_probabilities_np.shape}")

# --- 9. 제출 파일 생성 ---
if not os.path.exists(sample_submission_path):
    print(f"오류: 제출 샘플 파일 '{sample_submission_path}'를 찾을 수 없습니다.")
    exit()

submission_df_template = pd.read_csv(sample_submission_path)
submission_columns = list(submission_df_template.columns)
output_class_columns = [col for col in submission_columns if col != 'ID']

submission_data = {col: np.zeros(len(all_image_ids)) for col in output_class_columns}
submission_data['ID'] = all_image_ids # all_image_ids의 순서와 all_probabilities_np의 순서가 일치해야 함
final_submission_df = pd.DataFrame(submission_data)
final_submission_df = final_submission_df.set_index('ID')

# 대표 클래스 예측 확률을 원본 클래스 컬럼에 매핑
for i, img_id_tensor in enumerate(all_image_ids): # img_id_tensor가 실제 ID 문자열이 되도록 수정 필요
    img_id = img_id_tensor # DataLoader에서 문자열 ID를 그대로 반환했다면 OK
    if isinstance(img_id, torch.Tensor): # 만약 텐서로 반환되었다면 변환
        img_id = img_id.item() if img_id.numel() == 1 else str(img_id)


    for rep_class_idx, rep_class_name in enumerate(car_classnames): # car_classnames는 학습 시 사용된 대표 클래스 순서
        predicted_prob_for_rep_class = all_probabilities_np[i, rep_class_idx]

        if rep_class_name in representative_to_original_map:
            original_classes_for_this_rep = representative_to_original_map[rep_class_name]
            for original_class_name in original_classes_for_this_rep:
                if original_class_name in final_submission_df.columns:
                    final_submission_df.loc[img_id, original_class_name] = predicted_prob_for_rep_class
                # else: # 이 경고는 너무 많이 나올 수 있으므로 필요시 주석 해제
                #     print(f"경고: submission 파일에 '{original_class_name}' 컬럼이 없습니다. 확인 필요.")
        # else: # 이 경고도 너무 많이 나올 수 있음
        #     print(f"경고: 대표 클래스 '{rep_class_name}'에 대한 원본 클래스 매핑 정보가 없습니다.")

final_submission_df = final_submission_df.reset_index()

# 확률 정규화 및 클리핑
probs_to_normalize = final_submission_df[output_class_columns].values
prob_sums = probs_to_normalize.sum(axis=1, keepdims=True)
probabilities_normalized = np.divide(probs_to_normalize, prob_sums, out=np.full_like(probs_to_normalize, 1/len(output_class_columns)), where=prob_sums!=0) # 합이 0이면 균등 분배

y_pred_clipped = np.clip(probabilities_normalized, 1e-15, 1 - 1e-15)

final_submission_df[output_class_columns] = y_pred_clipped
final_submission_df = final_submission_df[submission_columns] # 원본 제출 파일과 동일한 컬럼 순서

submission_file_path = os.path.join(dataset_base_dir, 'submission_zeroshot.csv')
final_submission_df.to_csv(submission_file_path, index=False, encoding='utf-8')

print(f"\n제출 파일 생성 완료: {submission_file_path}")
print("생성된 제출 파일 미리보기 (상위 5개 행):")
print(final_submission_df.head())