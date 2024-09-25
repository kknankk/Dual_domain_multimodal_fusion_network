import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader,my_collate
import numpy as np
# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ECGModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=7,dropout=0.2):
        super(ECGModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x=torch.transpose(x,2,1)
        # print(f'x shape {x.shape}')
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 超参数
input_size = 12  # 12个导联
hidden_size = 64  # LSTM隐层大小
num_layers = 2  # LSTM层数
num_classes = 7  # 输出类别数
num_epochs = 70  # 训练轮数
batch_size = 64  # 批次大小
learning_rate = 0.001  # 学习率

# 假设 train_loader 是你的数据加载器
# train_loader = ...


# 实例化模型并将其移至 GPU
model = ECGModel(input_size, hidden_size, num_layers, num_classes).to(device)
# for name, parms in model.named_parameters():
# 	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.7,4.1,2.87,1.69,4.56,5.58,10.36]).to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader,val_loader=get_data_loader(batch_size=32)
train_ds,val_ds,test_ds=get_ECG_datasets()
# 训练模型
# all_train_labels = []
# all_train_outputs = []
output_file = '/data/ke/MIMIC_subset/result_ECG_temporal.txt'

with open(output_file, 'w') as f:
    f.write('epoch, train AUC, ' + ', '.join(f'class {i + 1} AUC' for i in range(num_classes)) + '\n')

    for epoch in range(num_epochs):
        epoch_loss=0
        all_train_labels = []
        all_train_outputs = []
        model.train()  # 设置模型为训练模式
        for inputs, labels in train_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)  # 转换为张量并移至 GPU
            labels = torch.tensor(labels, dtype=torch.float32).to(device)  # 转换为张量并移至 GPU

            optimizer.zero_grad()  # 清零梯度
            # print(f'input {inputs}')
            outputs = model(inputs)  # 前向传播





            
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_loss+=loss
            optimizer.step()  # 更新参数
            # print(f'Type of labels.cpu(): {type(labels.cpu())}')
            all_train_labels.append(labels.cpu())  # 将标签移回 CPU

            all_train_outputs.append(torch.sigmoid(outputs).detach().cpu())  # 移回 CPU并保存输出
            # print(f'train outputs with sigmoid {torch.sigmoid(outputs)} ')


        all_train_labels = torch.cat(all_train_labels).numpy()
        all_train_outputs = torch.cat(all_train_outputs).numpy()
        # print("Outputs NaN:", np.isnan(all_train_outputs).any())
        # print("Labels NaN:", np.isnan(all_train_labels).any())
        # print(f'output {all_train_outputs}')
        class_train_auc = []
        for i in range(num_classes):
            auc = roc_auc_score(all_train_labels[:, i], all_train_outputs[:, i])
            class_train_auc.append(auc)
            print(f'Class {i} - AUC: {auc:.4f}')

        auroc = roc_auc_score(all_train_labels, all_train_outputs, average='macro')
        print(f'train AUROC: {auroc:.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item()/(len(train_loader)):.4f}')
        f.write(f'{epoch + 1}, {auroc:.4f}, ' + ', '.join(f'{auc:.4f}' for auc in class_train_auc) + '\n')


    print(f'start eval--------------------------')
    # 计算AUROC
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)  # 转换为张量并移至 GPU
            labels = torch.tensor(labels, dtype=torch.float32).to(device)  # 转换为张量并移至 GPU

            outputs = model(inputs)  # 前向传播
            all_labels.append(labels.cpu())  # 将标签移回 CPU
            all_outputs.append(torch.sigmoid(outputs).detach().cpu())  # 移回 CPU并保存输出
            # print(f'train outputs with sigmoid {torch.sigmoid(outputs)} ')

    # 将列表合并为数组
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs)

    val_cls_auc=[]
    for i in range(num_classes):
        auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        val_cls_auc.append(auc)
        print(f'Class {i} - AUC: {auc:.4f}')
    # 计算AUROC
    auroc = roc_auc_score(all_labels, all_outputs, average='macro')
    print(f'val AUROC: {auroc:.4f}')
    f.write(f'start eval--------------------------'+'\n')
    f.write(f'{epoch + 1}, {auroc:.4f}, ' + ', '.join(f'{auc:.4f}' for auc in class_train_auc) + '\n')

