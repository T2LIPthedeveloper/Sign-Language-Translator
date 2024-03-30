import torch
from torch import nn

class XConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XConv, self).__init__()
        # This should include the operations for dynamically weighting and ordering points.
        # For simplicity, this uses a standard convolutional layer as a placeholder.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        # x: Input data of shape (batch_size, in_channels, num_points, 1)
        # Apply convolution
        x = self.conv(x)
        return x

class PointCNN(nn.Module):
    def __init__(self, num_classes):
        super(PointCNN, self).__init__()
        
        self.xconv1 = XConv(3, 64) 
        self.xconv2 = XConv(64, 128)
        self.xconv3 = XConv(128, 256)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Assuming x is of shape (batch_size, 3, num_points, 1) where 3 is for XYZ coordinates.
        # x = x.permute(2, 0, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        
        x = self.xconv1(x)
        x = self.xconv2(x)
        x = self.xconv3(x)
        
        # Apply global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def loss_fn(self, preds, targets):
        ce = nn.CrossEntropyLoss()
        ce_loss = ce(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return ce_loss, acc
