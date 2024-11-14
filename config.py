import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DIR = "C:\\development\\KinFaceW-I\\KinFaceW-I\\images"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_P = "gen_p.pth.tar"  # 부모에서 자식으로 변환하는 생성기
CHECKPOINT_GEN_C = "gen_c.pth.tar"  # 자식에서 부모로 변환하는 생성기
CHECKPOINT_CRITIC_P = "critic_p.pth.tar"  # 부모-자식 판별기
CHECKPOINT_CRITIC_C = "critic_c.pth.tar"  # 자식-부모 판별기

transforms = A.Compose(
    [
        A.Resize(width=256, height=256), 
        A.HorizontalFlip(p=0.5),         
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),  # PyTorch 텐서로 변환
    ],
    additional_targets={"image0": "image"},  
)
