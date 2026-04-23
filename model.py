import torch.nn as nn

class CNN_Transformer(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128*8,
            nhead=8
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(128*8, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0,3,1,2)
        x = x.reshape(b, w, c*h)

        x = self.transformer(x)
        x = self.fc(x)

        return x