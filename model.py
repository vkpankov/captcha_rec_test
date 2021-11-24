import torch.nn as nn
import utils

# Based on: https://github.com/dee1024/pytorch-captcha-recognition/blob/master/captcha_cnn_model.py
class CaptchaRecModel(nn.Module):
    def __init__(self):
        super(CaptchaRecModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(
                (utils.image_size[0] // 32)
                * (utils.image_size[1] // 32)
                * 1024,
                1024,
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.rfc = nn.Sequential(
            nn.Linear(1024, utils.captcha_len * utils.charset_len),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.layers(x)
        out = self.fc(out)
        out = self.rfc(out)
        return out
