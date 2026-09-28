from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 8,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_model_size_kb(
    model: nn.Module,
    bytes_per_parameter: int = 4,
) -> float:
    return (
        count_parameters(model)
        * bytes_per_parameter
        / 1024.0
    )


if __name__ == "__main__":
    model = SmallCNN(num_classes=4)

    dummy = torch.zeros(
        1,
        1,
        64,
        197,
    )

    output = model(dummy)

    print("P4 SmallCNN")
    print(f"Input shape:          {tuple(dummy.shape)}")
    print(f"Output shape:         {tuple(output.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(
        f"Estimated FP32 size:  "
        f"{estimate_model_size_kb(model):.2f} KB"
    )