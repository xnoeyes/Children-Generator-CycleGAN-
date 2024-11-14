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
from torch_fidelity import calculate_metrics 

dataset = KinFaceDataset(root_dir=config.TRAIN_DIR, transform=config.transforms)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


gen_C = Generator(input_channels=3, condition_dim=4, num_residuals=9).to(config.DEVICE)
opt_gen = optim.Adam(
    gen_C.parameters(),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999)
)


load_checkpoint(config.CHECKPOINT_GEN_C, gen_C, optimizer=None, lr=config.LEARNING_RATE)

def save_some_examples(gen_C, val_loader, folder):
    os.makedirs(folder, exist_ok=True)

    loop = tqdm(val_loader, leave=True)
    for idx, (parent_image, child_image, parent_gender, child_gender) in enumerate(loop):
        parent_image = parent_image.to(config.DEVICE) 
        child_image = child_image.to(config.DEVICE) 
        parent_gender = parent_gender.to(config.DEVICE)  # 성별 정보를 GPU로 이동
        child_gender = child_gender.to(config.DEVICE)

        gen_C.eval()  
        with torch.no_grad():
            fake_child = gen_C(parent_image, parent_gender, child_gender) 
            fake_child = fake_child * 0.5 + 0.5 
            
            save_image(fake_child, os.path.join(folder, f"fake_child_{idx}.png"))  
            save_image(child_image * 0.5 + 0.5, os.path.join(folder, f"child_{idx}.png"))  
        
        gen_C.train()  

if __name__ == "__main__":
    save_some_examples(gen_C, val_loader, folder='evaluation')  
    metrics = calculate_metrics(input1='evaluation', input2='evaluation', cuda=True, isc=True, fid=True, kid=True, kid_subset_size=321, verbose=True)
    print(metrics)
