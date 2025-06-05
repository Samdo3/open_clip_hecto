import os
import shutil

def merge_and_rename_files(base_dir, representative_class, source_classes_to_merge):
    """
    여러 소스 클래스 폴더의 파일들을 하나의 대표 클래스 폴더로 병합하고 파일명을 순차적으로 변경합니다.

    Args:
        base_dir (str): 'train' 폴더의 상위 경로 (예: './data/')
        representative_class (str): 파일들을 병합할 대표 클래스 폴더명
        source_classes_to_merge (list): 병합될 원본 소스 클래스 폴더명 리스트
    """
    representative_folder_path = os.path.join(base_dir, 'train', representative_class)

    # 대표 폴더가 없으면 생성
    if not os.path.exists(representative_folder_path):
        os.makedirs(representative_folder_path)
        print(f"대표 폴더 생성: {representative_folder_path}")
        current_file_count = 0
    else:
        # 대표 폴더 내 기존 파일 개수 확인 (파일명에서 숫자 부분 추출하여 최대값 찾기)
        existing_files = [f for f in os.listdir(representative_folder_path) if f.startswith(representative_class) and f.endswith('.jpg')]
        if not existing_files:
            current_file_count = 0
        else:
            max_num = -1
            for f_name in existing_files:
                try:
                    # 파일명_XXXX.jpg 형식에서 숫자 부분 추출
                    num_str = f_name.replace(representative_class + '_', '').replace('.jpg', '')
                    if num_str.isdigit():
                        num = int(num_str)
                        if num > max_num:
                            max_num = num
                except ValueError:
                    # 숫자 변환 실패 시 (예: 파일명 형식이 다를 경우)
                    print(f"경고: 파일명에서 숫자 추출 실패 - {f_name}")
                    continue
            current_file_count = max_num + 1
        print(f"대표 폴더 '{representative_class}'에 이미 파일이 존재합니다. 파일명은 {current_file_count}부터 시작합니다.")


    for source_class in source_classes_to_merge:
        source_folder_path = os.path.join(base_dir, 'train', source_class)

        if not os.path.exists(source_folder_path):
            print(f"경고: 소스 폴더를 찾을 수 없습니다 - {source_folder_path}")
            continue

        print(f"\n'{source_class}' 폴더의 파일들을 '{representative_class}' 폴더로 병합 시작...")

        image_files = sorted([f for f in os.listdir(source_folder_path) if f.endswith('.jpg')])

        for img_file in image_files:
            old_file_path = os.path.join(source_folder_path, img_file)

            # 새 파일명 생성: 대표클래스명_XXXX.jpg
            new_file_name = f"{representative_class}_{str(current_file_count).zfill(4)}.jpg"
            new_file_path = os.path.join(representative_folder_path, new_file_name)

            try:
                # 파일 이동 및 이름 변경 (shutil.move 사용)
                shutil.move(old_file_path, new_file_path)
                print(f"이동 및 이름 변경: '{old_file_path}' -> '{new_file_path}'")
                current_file_count += 1
            except Exception as e:
                print(f"오류 발생 (파일 이동/이름 변경 중): {old_file_path} -> {new_file_path}, 오류: {e}")

        # 원본 소스 폴더가 비었으면 삭제 (선택 사항)
        if not os.listdir(source_folder_path):
            try:
                os.rmdir(source_folder_path)
                print(f"소스 폴더 삭제 (비어 있음): {source_folder_path}")
            except Exception as e:
                print(f"오류 발생 (소스 폴더 삭제 중): {source_folder_path}, 오류: {e}")
        else:
            print(f"경고: 소스 폴더 '{source_folder_path}'에 삭제되지 않은 파일/폴더가 남아있습니다.")

    print(f"\n'{representative_class}' 폴더로의 병합 작업 완료. 총 {current_file_count}개의 이미지가 있습니다.")


# --- 설정 ---
dataset_base_dir = './hecto_data' # 현재 스크립트가 있는 위치를 기준으로 dataset 폴더가 있다고 가정

# 동일 클래스로 처리할 목록 (대표 클래스명, [병합될 소스 클래스명 리스트])

# 1) 'K5_3세대_하이브리드_2020_2022' (대표) <- 'K5_하이브리드_3세대_2020_2023' (소스)
class_map_1 = {
    "representative": "K5_3세대_하이브리드_2020_2022",
    "sources_to_merge": ["K5_하이브리드_3세대_2020_2023"]
}

# 2) '디_올뉴니로_2022_2025' (대표) <- '디_올_뉴_니로_2022_2025' (소스)
class_map_2 = {
    "representative": "디_올뉴니로_2022_2025",
    "sources_to_merge": ["디_올_뉴_니로_2022_2025"]
}

# 3) '718_박스터_2017_2024' (대표) <- '박스터_718_2017_2024' (소스)
class_map_3 = {
    "representative": "718_박스터_2017_2024",
    "sources_to_merge": ["박스터_718_2017_2024"]
}

# 4) 'RAV4_2016_2018' (대표) <- '라브4_4세대_2013_2018' (소스)
class_map_4 = {
    "representative": "RAV4_2016_2018",
    "sources_to_merge": ["라브4_4세대_2013_2018"]
}

# 5) 'RAV4_5세대_2019_2024' (대표) <- '라브4_5세대_2019_2024' (소스)
class_map_5 = {
    "representative": "RAV4_5세대_2019_2024",
    "sources_to_merge": ["라브4_5세대_2019_2024"]
}

all_class_maps = [class_map_1, class_map_2, class_map_3, class_map_4, class_map_5]

# --- 실행 ---
if __name__ == "__main__":
    if not os.path.exists(os.path.join(dataset_base_dir, 'train')):
        print(f"오류: '{os.path.join(dataset_base_dir, 'train')}' 폴더를 찾을 수 없습니다. dataset_base_dir 경로를 확인하세요.")
    else:
        for class_map_info in all_class_maps:
            print(f"\n================== {class_map_info['representative']} 병합 작업 시작 ==================")
            merge_and_rename_files(dataset_base_dir, class_map_info["representative"], class_map_info["sources_to_merge"])
            print(f"================== {class_map_info['representative']} 병합 작업 종료 ==================\n")
        print("모든 지정된 클래스 병합 작업이 완료되었습니다.")