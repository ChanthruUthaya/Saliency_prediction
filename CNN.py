from global_vars import *

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )

        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv2)

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv3)

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.fc1 = nn.Linear(11*11*128, 4608)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)

        # self.norm1 = nn.BatchNorm2d(32)
        # self.norm2 = nn.BatchNorm2d(64)
        # self.norm3 = nn.BatchNorm2d(128)
        # self.norm4 = nn.BatchNorm1d(4608)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu((self.conv1(images)))
        x = self.pool1(x)

        x = F.relu((self.conv2(x)))
        x = self.pool2(x)

        x = F.relu((self.conv3(x)))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        # x = self.norm4(x)

        # maxout
        # slice the tensor in half
        slice1 = x[:,:2304]
        slice2 = x[:,2304:]

        x_max = torch.max(slice1, slice2)

        x = self.fc2(x_max)


        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
