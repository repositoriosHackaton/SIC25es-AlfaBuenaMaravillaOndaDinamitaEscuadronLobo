#!/usr/bin/env python3
# modelo_pointnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class STNkd(nn.Module):
    
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
  
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
  
        x, _ = torch.max(x, 2)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
    
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batchsize, 1)
        x = x + iden  
        x = x.view(-1, self.k, self.k)
        return x

class PointNetClasificador(nn.Module):
    def __init__(self, k=3, num_classes=29):
        
        super(PointNetClasificador, self).__init__()
        self.stn = STNkd(k=k)
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        
        trans = self.stn(x)
        x = torch.bmm(trans, x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
    
        x, _ = torch.max(x, 2)  
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  
        return x

def test():
    
    model = PointNetClasificador(k=3, num_classes=29)
    points = torch.randn(8, 3, 21)  
    out = model(points)
    print("Salida shape:", out.shape)  

if __name__ == "__main__":
    test()
