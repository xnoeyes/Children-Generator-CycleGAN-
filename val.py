import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import load_checkpoint
import config
from dataset import KinFaceDataset  # KinFaceDataset 사용
from generator import Generator
from torchvision.utils import save_image
from tqdm import tqdm
from torch_fidelity import calculate_metrics  # 성능 평가를 위한 라이브러리 추가

# 전체 데이터셋 로드
dataset = KinFaceDataset(root_dir=config.TRAIN_DIR, transform=config.transforms)

# 데이터셋을 학습 및 검증으로 분할
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 검증 데이터 로드
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 생성기 초기화
gen_C = Generator(input_channels=3, condition_dim=4, num_residuals=9).to(config.DEVICE)
opt_gen = optim.Adam(
    gen_C.parameters(),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999)
)

# 체크포인트에서 생성기 로드
load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, optimizer=None, lr=config.LEARNING_RATE)

def save_some_examples(gen_C, val_loader, folder):
    # 저장할 디렉토리가 존재하는지 확인하고, 없으면 생성
    os.makedirs(folder, exist_ok=True)

    loop = tqdm(val_loader, leave=True)
    for idx, (parent_image, child_image, parent_gender, child_gender) in enumerate(loop):
        parent_image = parent_image.to(config.DEVICE) 
        child_image = child_image.to(config.DEVICE) 
        parent_gender = parent_gender.to(config.DEVICE)  # 성별 정보를 GPU로 이동
        child_gender = child_gender.to(config.DEVICE)

        gen_C.eval()  # 생성기 평가 모드
        with torch.no_grad():
            fake_child = gen_C(parent_image, parent_gender, child_gender)  # 성별 정보 전달
            fake_child = fake_child * 0.5 + 0.5  # 정규화 제거
            
            # 가짜 자식 이미지 및 원본 자식 이미지 저장
            save_image(fake_child, os.path.join(folder, f"fake_child_{idx}.png"))  
            save_image(child_image * 0.5 + 0.5, os.path.join(folder, f"child_{idx}.png"))  
        
        gen_C.train()  # 생성기 학습 모드로 전환

if __name__ == "__main__":
    save_some_examples(gen_C, val_loader, folder='evaluation')  # 예시 이미지 저장

    # 성능 평가
    metrics = calculate_metrics(input1='evaluation', input2='evaluation', cuda=True, isc=True, fid=True, kid=True, kid_subset_size=321, verbose=True)

    # 평가 결과 출력
    print(metrics)
