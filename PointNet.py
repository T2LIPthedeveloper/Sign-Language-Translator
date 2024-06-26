import torch
from torch import nn

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        #self.dropout = nn.Dropout(0.2)
        # Shared MLP (multi-layer perceptron)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Global max pooling
        self.global_max_pool = nn.MaxPool1d(21)


        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        #self.dropout()
        # x: input point cloud tensor of shape (batch_size, num_points, num_channels)

        # Permute the tensor to have shape (batch_size, num_channels, num_points)
        x = x.permute(0, 2, 1)

        # Apply shared MLP
        x = self.mlp(x)

        # Global max pooling
        x = self.global_max_pool(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        return x
    
    def loss_fn(self, preds, targets):
        ce = nn.CrossEntropyLoss()
        ce_loss = ce(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return ce_loss, acc