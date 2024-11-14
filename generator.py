import torch
import torch.nn as nn

# ConvBlock 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # 합성곱 레이어와 인스턴스 정규화, 활성화 함수를 포함한 순차적 모듈을 정의
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down #nn.Conv2d를 사용하여 일반적인 합성곱을 수행
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        # 입력을 정의된 합성곱 블록에 통과
        return self.conv(x)

# ResidualBlock 클래스 정의: ResNet블록
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 두 개의 ConvBlock을 포함한 순차적 모듈을 정의
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1), #마지막 층 활성화함수 x 
        )

    def forward(self, x):
        # 입력에 잔차 블록의 출력을 더하여 반환
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, condition_dim=4, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        self.condition_dim = condition_dim
        
        # 초기 합성곱 레이어
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + condition_dim, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        # 다운샘플링 블록
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
            ]
        )
        
        # 잔차 블록
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # 업샘플링 블록
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        
        # 마지막 합성곱 레이어
        self.last = nn.Conv2d(num_features, input_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, parent_img, parent_gender, child_gender):
        batch_size = parent_img.size(0)
        img_size = parent_img.size(2)
        
        #초기에는 1x1로 설정하고, 이후 expand를 통해 이미지의 크기에 맞게 확장 
        condition = torch.cat([parent_gender, child_gender], dim=1).view(batch_size, self.condition_dim, 1, 1) 
        condition = condition.expand(batch_size, self.condition_dim, img_size, img_size)
        
        # 입력 이미지와 조건을 결합
        combined_input = torch.cat([parent_img, condition], dim=1)
        
        # 네트워크 통과
        x = self.initial(combined_input)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        
        return torch.tanh(self.last(x))
    
# Generator 
# class Generator(nn.Module):
#     def __init__(self, img_channels, num_features = 64, num_residuals=9):
#         super().__init__()
#         # 초기 합성곱 레이어
#         self.initial = nn.Sequential(
#             nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
#             nn.InstanceNorm2d(num_features),
#             nn.ReLU(inplace=True),
#         )
#         # 다운샘플링 블록
#         self.down_blocks = nn.ModuleList(
#             [
#                 ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
#                 ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
#             ]
#         )
#         # 잔차 블록들을 정의
#         self.res_blocks = nn.Sequential(
#             *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
#         )
#         # 업샘플링 블록을 정의
#         self.up_blocks = nn.ModuleList(
#             [
#                 ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
#             ]
#         )
#         # 마지막 합성곱 레이어를 정의
#         self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

#     def forward(self, x):
#         # 입력을 초기 블록에 통과
#         x = self.initial(x)
#         # 다운샘플링 블록을 통해 입력을 통과
#         for layer in self.down_blocks:
#             x = layer(x)
#         # 잔차 블록을 통해 입력을 통과
#         x = self.res_blocks(x)
#         # 업샘플링 블록을 통해 입력을 통과
#         for layer in self.up_blocks:
#             x = layer(x)
#         # 마지막 레이어를 통과시키고 tanh를 적용하여 반환
#         return torch.tanh(self.last(x))

# 테스트 함수 
#def test():
#    img_channels = 3
#    img_size = 256
#   x = torch.randn((2, img_channels, img_size, img_size))  # 임의의 입력 생성
#    gen = Generator(img_channels, 9)  # 생성기 객체 생성
#    print(gen(x).shape)  # 생성기의 출력 크기를 출력

#if __name__ == "__main__":
#    test()  # 테스트 함수 실행
