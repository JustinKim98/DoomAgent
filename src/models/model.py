import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box

class PolicyModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim):
        super(PolicyModel, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]

        base_channel_size = 32

        # vzd.Button.ALTATTACK,
        self.conv2d1 = torch.nn.Conv2d(input_channels, base_channel_size, kernel_size=4)
        self.bn1 = torch.nn.BatchNorm2d(base_channel_size)
        self.maxpool1 = torch.nn.MaxPool2d(4, 2)
        self.relu1 = torch.nn.LeakyReLU()

        self.conv2d2 = torch.nn.Conv2d(
            base_channel_size, base_channel_size * 2, kernel_size=4
        )
        self.bn2 = torch.nn.BatchNorm2d(base_channel_size * 2)
        self.maxpool2 = torch.nn.MaxPool2d(4, 2)
        self.relu2 = torch.nn.LeakyReLU()

        self.conv2d3 = torch.nn.Conv2d(
            base_channel_size * 2, base_channel_size * 4, kernel_size=4
        )
        self.bn3 = torch.nn.BatchNorm2d(base_channel_size * 4)
        self.maxpool3 = torch.nn.MaxPool2d(4, 2)
        self.relu3 = torch.nn.LeakyReLU()

        self.conv2d4 = torch.nn.Conv2d(
            base_channel_size * 4, base_channel_size * 8, kernel_size=4
        )
        self.bn4 = torch.nn.BatchNorm2d(base_channel_size * 8)
        self.maxpool4 = torch.nn.MaxPool2d(4, 2)
        self.relu4 = torch.nn.LeakyReLU()

        print(f"observation_space shape : {observation_space.shape}")
        self.flatten = torch.nn.Flatten()
        
        dim_size = 1024 if observation_space.shape[0] > 12 else 4096

        self.linear1 = torch.nn.Linear(38400, dim_size)
        self.relu4 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(dim_size, features_dim)
        self.relu5 = torch.nn.LeakyReLU()

    def forward(self, state_input: torch.Tensor):
        h1 = self.conv2d1(state_input)
        h1 = self.bn1(h1)
        h1 = self.maxpool1(h1)
        h1 = self.relu1(h1)

        h2 = self.conv2d2(h1)
        h2 = self.bn2(h2)
        h2 = self.maxpool2(h2)
        h2 = self.relu2(h2)

        h3 = self.conv2d3(h2)
        h3 = self.bn3(h3)
        h3 = self.maxpool3(h3)
        h3 = self.relu3(h3)

        h4 = self.conv2d4(h3)
        h4 = self.bn4(h4)
        h4 = self.maxpool4(h4)
        h4 = self.relu4(h4)

        h5 = self.flatten(h4)
        h5 = self.linear1(h5)
        h5 = self.relu4(h5)

        h6 = self.linear2(h5)
        out = self.relu5(h6)
        return out
