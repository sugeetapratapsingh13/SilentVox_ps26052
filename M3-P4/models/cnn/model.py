import torch
import torch.nn as nn

class SpeechEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    model = SpeechEnhancementCNN()

    x = torch.randn(1, 1, 257, 100)
    y = model(x)

    params = sum(p.numel() for p in model.parameters())

    print("CNN model test")
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameters  :", params)
