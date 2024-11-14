import os
import torch
import config
import torch.nn as nn
import torch.optim as optim
from dataset import KinFaceDataset  # KinFaceDataset을 임포트
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator

def train_func(disc_P, disc_C, gen_P, gen_C, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, loader, epoch):
    loop = tqdm(loader, leave=True)  
    os.makedirs('saved_images', exist_ok=True)

    for idx, (parent_image, child_image, parent_gender, child_gender) in enumerate(loop):
        parent_image = parent_image.to(config.DEVICE)  # 부모 이미지를 장치로 이동
        child_image = child_image.to(config.DEVICE)  # 자식 이미지를 장치로 이동
        parent_gender = parent_gender.to(config.DEVICE)  # 부모 성별 정보 장치로 이동
        child_gender = child_gender.to(config.DEVICE)  # 자식 성별 정보 장치로 이동

        # 판별기 P 및 C 학습
        with torch.autocast(device_type="cuda" if config.DEVICE.type == 'cuda' else "cpu"):  
            fake_child = gen_C(parent_image, parent_gender, child_gender)  # 부모 이미지를 기반으로 가짜 자식 이미지 생성

            D_P_real = disc_P(child_image, parent_gender, child_gender)  # 진짜 자식 이미지를 판별기 P로 전달하여 판별
            D_P_fake = disc_P(fake_child.detach(), parent_gender, child_gender)  # 생성된 가짜 자식 이미지를 판별기 P로 전달하여 판별
            
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))  # 진짜 자식 이미지에 대한 손실 계산
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))  # 가짜 자식 이미지에 대한 손실 계산
            
            D_P_loss = D_P_real_loss + D_P_fake_loss  # 판별기 P에 대한 총 손실 계산 

            fake_parent = gen_P(child_image, child_gender, parent_gender)  # 자식 이미지를 기반으로 가짜 부모 이미지 생성 (성별 벡터 포함)
            D_C_real = disc_C(parent_image, parent_gender, child_gender)  # 진짜 부모 이미지를 판별기 C로 전달하여 판별
            D_C_fake = disc_C(fake_parent.detach(), parent_gender, child_gender)  # 생성된 가짜 부모 이미지를 판별기 C로 전달하여 판별 

            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))  # 진짜 부모 이미지에 대한 손실 계산
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))  # 가짜 부모 이미지에 대한 손실 계산

            D_C_loss = D_C_real_loss + D_C_fake_loss  # 판별기 C에 대한 총 손실 계산

            D_loss = (D_P_loss + D_C_loss) / 2  # 평균 손실 계산
        
        opt_disc.zero_grad()  # 판별기 옵티마이저의 그래디언트 초기화
        d_scaler.scale(D_loss).backward()  # 손실에 대한 그래디언트 계산 및 스케일링
        d_scaler.step(opt_disc)  # 옵티마이저 스텝 수행
        d_scaler.update()  # 스케일러 업데이트

        # 생성기 P 및 C 학습
        with torch.autocast(device_type="cuda" if config.DEVICE.type == 'cuda' else "cpu"):
            D_P_fake = disc_P(fake_child.detach(), parent_gender, child_gender)  # 가짜 자식 이미지 판별
            D_C_fake = disc_C(fake_parent.detach(), parent_gender, child_gender)  # 가짜 부모 이미지 판별
            loss_G_C = mse(D_P_fake, torch.ones_like(D_P_fake))  # 생성기 손실 계산
            loss_G_P = mse(D_C_fake, torch.ones_like(D_C_fake))  # 생성기 손실 계산
            
            # 주기 손실 (Cycle Consistency Loss)
            cycle_parent = gen_P(fake_child, child_gender, parent_gender)  # 자식에서 부모로 복원
            cycle_child = gen_C(fake_parent, parent_gender, child_gender)  # 부모에서 자식으로 복원
            cycle_parent_loss = L1(parent_image, cycle_parent)  # 부모 주기 손실
            cycle_child_loss = L1(child_image, cycle_child)  # 자식 주기 손실

            G_loss = (
                loss_G_C
                + loss_G_P
                + cycle_parent_loss * config.LAMBDA_CYCLE  # 부모 주기 손실 가중치 적용
                + cycle_child_loss * config.LAMBDA_CYCLE  # 자식 주기 손실 가중치 적용
            )
        
        opt_gen.zero_grad()  # 생성기 옵티마이저의 그래디언트 초기화
        g_scaler.scale(G_loss).backward()  # 손실에 대한 그래디언트 계산 및 스케일링
        g_scaler.step(opt_gen)  # 옵티마이저 스텝 수행
        g_scaler.update()  # 스케일러 업데이트

        print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] [Batch {idx+1}/{len(loader)}] "
              f"[D loss: {D_loss.item():.4f}] [G loss: {G_loss.item():.4f}]")
        
        if epoch % 10 == 0:
            save_image(parent_image * 0.5 + 0.5, f"saved_images/parent_epoch_{epoch}.png")
            save_image(fake_child * 0.5 + 0.5, f"saved_images/fake_child_epoch_{epoch}.png")
            save_image(child_image * 0.5 + 0.5, f"saved_images/child_epoch_{epoch}.png")
            save_image(fake_parent * 0.5 + 0.5, f"saved_images/fake_parent_epoch_{epoch}.png")

def main():
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 장치 설정을 메인 함수에서 수행
    disc_P = Discriminator(input_channels=3, condition_dim=4).to(config.DEVICE)
    disc_C = Discriminator(input_channels=3, condition_dim=4).to(config.DEVICE)
    gen_P = Generator(input_channels=3, condition_dim=4, num_residuals=9).to(config.DEVICE)
    gen_C = Generator(input_channels=3, condition_dim=4, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_P.parameters()) + list(gen_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    L1 = nn.L1Loss()  # L1 손실 함수 정의
    mse = nn.MSELoss()  # MSE 손실 함수 정의

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_P, disc_P, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_C, disc_C, opt_disc, config.LEARNING_RATE)
    
    # KinFaceDataset을 사용하여 데이터셋 로드
    dataset = KinFaceDataset(root_dir=config.TRAIN_DIR, transform=config.transforms)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = config.BATCH_SIZE
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    g_scaler = torch.amp.GradScaler()  # 생성기 그래디언트 스케일러 설정
    d_scaler = torch.amp.GradScaler()  # 판별기 그래디언트 스케일러 설정


    for epoch in range(config.NUM_EPOCHS):
        train_func(disc_P, disc_C, gen_P, gen_C, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, train_loader, epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)
            save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_CRITIC_P)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_C)


if __name__ == "__main__":
    main() 
