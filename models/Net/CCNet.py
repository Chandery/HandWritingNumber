import torch.nn as nn
import torch

class CCNet(nn.Module):
    def __init__(self):
        super(CCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 18, kernel_size=3, padding=1)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(288,10)
        self.relu4 = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) # *label.shape = (bs, classnumber) 所以要保证所有类别加起来为1，因此dim=1

    def forward(self, x):
        x = self.relu1(self.MaxPool1(self.conv1(x)))
        
        x = self.relu2(self.MaxPool2(self.conv2(x)))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu3(self.MaxPool3(x))

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu4(self.fc(x))

        x = self.softmax(x)

        return x
    
if __name__ == '__main__':
    model = CCNet()
    X = torch.randn(32, 1, 28, 28)
    for layer in model.children():
        X = layer(X)
        print("Layer: ", layer.__class__.__name__, 'Output shape: \t', X.shape)