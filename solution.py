import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),            
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3),            
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
            )

        self.flatten = nn.Flatten()
        self.enc_lin = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),            
            nn.Linear(32, 1)
            )
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.enc_conv(x)
        x = self.flatten(x)
        x = self.enc_lin(x)
        x = self.act(x)
        
        return x


class OpenEyesClassificator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = ConvNet()
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.to(self.device)
        self.model.eval()
        self.test_transforms = transforms.Compose([
            transforms.Resize((24, 24)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def predict(self, inpIm):
        image = Image.open(inpIm)
        image_tensor = self.test_transforms(image).float()
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(self.device)
        is_open_score = self.model(image_tensor).item()
        
        return is_open_score


if __name__ == "__main__":
    model = ConvNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clf = OpenEyesClassificator(device)
    score = clf.predict('/home/olga/Desktop/Eyes/Screenshot from 2021-11-14 20-10-20.png')
    print('is_open_score:', score)