import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/data/ke/MIMIC_subset'))
# print(a)
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
import numpy as np
from model.ECG_model import LSTM


def main():


    train_dl, val_dl = get_data_loader(batch_size=16)

    # 定义模型
    model = LSTM(input_dim=12, num_classes=7,  dropout=0.2, layers=1)

    model.train()

    # 损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    criterion =nn.BCELoss()#need add sigmod on model
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(90):
        train_loss = train(train_dl,model,criterion,optimizer)  
        # val_loss,val_acc = test()  


def train(train_dl,model,criterion,optimizer):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_dl):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        targets = torch.from_numpy(targets).float()
        # torch.from_numpy(y_ehr).float()
        # 前向传播
        pred = model(data)  # 数据传入模型
        # print(f'output {pred}')#first output are all 0.5 
        # print(f'output size {pred.shape}')#[batch_size,num_cls]

        # 计算损失
        loss = criterion(pred, targets)
        # prec1=accuracy(pred.data,targets.data)
        #TODO: add AUROC for acc
        #TODO:查看不同epoch的computeAUROC是cat不同epoch之后的结果吗？
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
        # return (loss.avg, pred.avg)
        return loss


if __name__ == '__main__':
    main()




