import torch
import torch.nn as nn

class UnetGenerator(nn.Module):
    '''UNet Generator model for minifluor EEM translation.'''
    def __init__(self, in_channels=3, ngf=64):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, ngf, 4, 2, 1), nn.LeakyReLU(0.2)) # 256p --> 128p
        self.down_blocks = nn.ModuleList([
            Block(ngf, ngf * 2, downsample=True, activation="leaky"), # 128p --> 64p
            Block(ngf * 2, ngf * 4, downsample=True, activation="leaky"), # 64p --> 32p
            Block(ngf * 4, ngf * 8, downsample=True, activation="leaky"), # 32p --> 16p
            Block(ngf * 8, ngf * 8, downsample=True, activation="leaky"), # 16p --> 8p
            Block(ngf * 8, ngf * 8, downsample=True, activation="leaky"), # 8p --> 4p
            Block(ngf * 8, ngf * 8, downsample=True, activation="leaky"), # 4p --> 2p
        ])
        self.bottleneck = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1), nn.ReLU()) # 2p --> 1p
        self.up_blocks = nn.ModuleList([
            Block(ngf * 8, ngf * 8, downsample=False, activation="relu"), # 1p --> 2p
            Block(ngf * 8 * 2, ngf * 8, downsample=False, activation="relu"), # 2p --> 4p
            Block(ngf * 8 * 2, ngf * 8, downsample=False, activation="relu"), # 4p --> 8p
            Block(ngf * 8 * 2, ngf * 8, downsample=False, activation="relu"), # 8p --> 16p
            Block(ngf * 8 * 2, ngf * 4, downsample=False, activation="relu"), # 16p --> 32p
            Block(ngf * 4 * 2, ngf * 2, downsample=False, activation="relu"), # 32p --> 64p
            Block(ngf * 2 * 2, ngf, downsample=False, activation="relu"), # 64p --> 128p
        ])
        self.final_up = nn.Sequential(nn.ConvTranspose2d(ngf * 2, in_channels, 4, 2, 1), nn.Tanh()) # 128p --> 256p

    def forward(self, x):
        down_outputs = [self.initial_down(x)]
        for i, block in enumerate(self.down_blocks):
            down_outputs.append(block(down_outputs[-1]))
        
        up_output = self.bottleneck(down_outputs[-1])
        up_output = self.up_blocks[0](up_output)
        for i, block in enumerate(self.up_blocks[1:]):
            up_output = block(torch.cat([up_output, down_outputs[-i-1]], 1)) # add skip connection
        return self.final_up(torch.cat([up_output, down_outputs[0]], 1)) # add skip connection

class Block(nn.Module):
    '''A single block for constructing the UNet model.'''
    def __init__(self, in_channels, out_channels, downsample=True, activation="relu"):
        super().__init__()
        conv_layer = nn.Conv2d if downsample else nn.ConvTranspose2d
        self.conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2) # activation layer
        )

    def forward(self, x):
        return self.conv(x)