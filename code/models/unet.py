import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True):
        super().__init__()
        if down:  # encoder
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            self.block = nn.Sequential(*layers)
        else:     # decoder
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(3,   64, down=True,  use_bn=False)  # 32 -> 16
        self.enc2 = UNetBlock(64,  128, down=True)
        self.enc3 = UNetBlock(128, 256, down=True)
        self.enc4 = UNetBlock(256, 512, down=True)

        # Decoder
        self.dec1 = UNetBlock(512,     256, down=False)
        self.dec2 = UNetBlock(256+256, 128, down=False)
        self.dec3 = UNetBlock(128+128,  64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64+64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.enc4(e3)

        d1 = self.dec1(bottleneck)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        out = self.final(torch.cat([d3, e1], dim=1))
        return out
