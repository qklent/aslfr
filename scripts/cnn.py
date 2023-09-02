import torch
from torch import nn
# added skip connections

class SignRecognition(nn.Module):
    def __init__(self, frames, kernel_size=2, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(frames, frames // 2, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(frames // 2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(1)
        
        self.lin1 = nn.Linear(20, 128)
        self.lin2 = nn.Linear(128, 256)
        #self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        
        #self.skipln = nn.Linear(20, 256)
        
    def forward(self, x):
        x = self.conv1(x) # B, 64 41 1
        x = self.leaky_relu(self.bn1(x))

        x = x.view(x.shape[0], 1, 64, 41) 
        
    
        x = self.conv2(x) # B, 1, 32, 20
        x = self.leaky_relu(self.bn2(x))
        
        x = x.squeeze(1)
        
        # skip1 = self.skipln(x) # B 32 256

        x = self.lin1(x)
        x = self.leaky_relu(x)
        
        x = self.lin2(x)
        x = self.tanh(x)
        
        
        return x #+ skip1


if __name__ == "__main__":
    model = SignRecognition(128)
    x = torch.randn(1, 128, 82, 2)
    out = model(x)
    print(out.shape)
    print(out[0])