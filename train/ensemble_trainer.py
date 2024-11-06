import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
# 设置基础路径
base_path = Path("/home/mimic/MIMIC_subset/MIMIC_subset")
sys.path.append(str(base_path))

from sklearn.metrics import roc_auc_score, average_precision_score
from dataset.fusion_dataset import load_cxr_ecg_ds, get_ecgcxr_data_loader

# 假设你已经定义了这两个模型的类
from model.CXR_model import CXRModels
from model.ECG_model import ResNet1d
device_ids = [2, 3]
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 选择 GPU 2 作为主设备

# 加载模型
# def load_model(checkpoint_path, model_class):
#     model = model_class()
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     return model

def load_model(checkpoint_path, model_class):
    model = model_class()
    checkpoint = torch.load(checkpoint_path)

    # 处理 state_dict 的命名
    state_dict = checkpoint['state_dict']
    if 'module.' in list(state_dict.keys())[0]:  # 检查是否包含 'module.'
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 去除前缀

    model.load_state_dict(state_dict)
    model = model.to(device)  # 将模型移动到指定设备
    model.eval()
    return model


# 设置路径
cxr_model_path = '/home/mimic/MIMIC_subset/MIMIC_subset/checkpoints/cxr/S_T/best_checkpoint.pth.tar'
ecg_model_path = '/home/mimic/MIMIC_subset/MIMIC_subset/checkpoints/ecg/S_T/best_checkpoint.pth.tar'

# 加载模型
cxr_model = load_model(cxr_model_path, CXRModels)
ecg_model = load_model(ecg_model_path, ResNet1d)

cxr_model = torch.nn.DataParallel(cxr_model, device_ids=device_ids)
ecg_model = torch.nn.DataParallel(ecg_model, device_ids=device_ids)


# 加载数据
batch_size = 32  # 假设 args.batch_size 已定义
fusion_train_dl, fusion_val_dl = get_ecgcxr_data_loader(batch_size=batch_size)

# 初始化用于存储 logits 和标签的列表
all_logits = []
all_labels = []

# 推理过程
with torch.no_grad():
    for ecg_data, img, target, seq_length, pairs in fusion_val_dl:
        # 将数据移至 GPU（如果有可用 GPU）
        # ecg_data = ecg_data.float().to(device)  # device 应该是 'cuda' 或 'cpu'
        # if isinstance(ecg_data, list):
        #     ecg_data = torch.tensor(ecg_data).float().to(device)  # 转换为张量并移动到设备
        # else:
        #     ecg_data = ecg_data.float().to(device)  # 如果已经是张量，直接转换
        ecg_data_np = np.array(ecg_data)
        ecg_data = torch.from_numpy(ecg_data_np).float()
        ecg_data = ecg_data.float().to(device)
        target = torch.from_numpy(target).float()
        target = target.to(device)



        # 确保
        img = img.float().to(device)

        # 通过模型推理
        _,cx_logits = cxr_model(img)  # CXR 模型的输出
        # print(f'cx_logits{cx_logits.shape}')
        _,ecg_logits = ecg_model(ecg_data)  # ECG 模型的输出
        # print(f'ecg_logits,{ecg_logits.shape}')
        # 合并 logits
        
        # final_logits = cx_logits + ecg_logits
        final_logits = ecg_logits
        # print(f'final_logits {final_logits}')
        # print(f'final_logits {final_logits.shape}')
        # for a in final_logits:
        #     print(f'a {a.shape}')
        # final_logits = torch.cat(final_logits, dim=0)

        # 使用 sigmoid 将 logits 转换为概率
        probabilities = torch.sigmoid(final_logits)

        # 存储 logits 和目标标签
        all_logits.append(probabilities.cpu())
        all_labels.append(target.cpu())

# 将 logits 和标签拼接为一个张量
all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# 计算 AUROC 和 AUPRC
y_true = all_labels.numpy()
y_scores = all_logits.numpy()

# 计算 AUROC 和 AUPRC
auroc = roc_auc_score(y_true, y_scores, average='macro')
auprc = average_precision_score(y_true, y_scores, average='macro')

print(f'ecg AUROC: {auroc:.4f}')
print(f'ecg AUPRC: {auprc:.4f}')
