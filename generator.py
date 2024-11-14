import torch
import torch.nn as nn

# ConvBlock 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down 
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# ResidualBlock 
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1), #마지막 층 활성화함수 x 
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, condition_dim=4, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        self.condition_dim = condition_dim
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + condition_dim, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        self.last = nn.Conv2d(num_features, input_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, parent_img, parent_gender, child_gender):
        batch_size = parent_img.size(0)
        img_size = parent_img.size(2)
        
        condition = torch.cat([parent_gender, child_gender], dim=1).view(batch_size, self.condition_dim, 1, 1) 
        condition = condition.expand(batch_size, self.condition_dim, img_size, img_size)
        
        combined_input = torch.cat([parent_img, condition], dim=1)
        
        x = self.initial(combined_input)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        
        return torch.tanh(self.last(x))
    
