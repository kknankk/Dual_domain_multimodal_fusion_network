
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
from train.trainer_utils import Trainer
from torch.autograd import Variable
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
from dataset.cxr_dataset import get_cxr_datasets,get_cxrdata_loader
import numpy as np
# from model.ECG_model import LSTM, Spect_CNN, ECGModel, ResNet1d, spectrogram_model
from model.CXR_model import CXRModels, wavevit_s
from argument import args_parser
parser = args_parser()
args = parser.parse_args()
from dataset.fusion_dataset import get_ecgcxr_data_loader
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc,average_precision_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
train_dl, val_dl = get_ecgcxr_data_loader(batch_size=args.batch_size)
torch.set_printoptions(threshold=torch.inf, linewidth=1000)
# 设置可用 GPU 设备
# device_ids = [2,3, 4, 5, 6,7]  # GPU 索引从 0 开始，实际对应物理的第 4, 5, 6, 7 个 GPU
# torch.cuda.set_device(device_ids[0])  # 将默认 GPU 设为第一个

class Fusion_trainer(Trainer):
    def __init__(self, train_dl, val_dl, args, ecg_model, cxr_model):
        super(Fusion_trainer,self).__init__(args)
        self.args = args
        self.ecg_model = ecg_model
        self.cxr_model = cxr_model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epoch=0
        os.environ["CUDA_VISIBLE_DEVICES"] = " 0,1,2,3" #use first four GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化分类器
        # self.classifier = nn.Linear(22912, 7)
        # frequency 是960，7
        # self.classifier = nn.Sequential(
        #         nn.Linear(512, 7),  # 隐藏层
        #         # nn.ReLU(),              # 激活函数
        #         # nn.Linear(512, 7)       # 输出层
        #         )
        self.classifier = nn.Sequential(
        nn.Linear(512, 512),  # 添加一个隐藏层
        nn.ReLU(),            # 激活函数
        nn.Dropout(p=0.5),    # Dropout层，丢弃50%的神经元
        nn.Linear(512, 7)     # 输出层
    )

        # self.criterion = nn.BCEWithLogitsLoss()  # 可以根据任务修改损失函数
        # self.optimizer = optim.Adam(list(ecg_model.parameters()) + list(cxr_model.parameters()) + list(self.classifier.parameters()), lr=args.lr)
        # self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.7,3.1,1.9,0.7,3.6,4.6,9.4]).to(self.device)) 
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.7,3.1,1.8,0.68,3.5,4.5,9.3]).to(self.device)) 

        # self.optimizer=optim.Adam(self.model.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-3,)
        self.optimizer = optim.Adam(
            list(self.ecg_model.parameters()) + 
            list(self.cxr_model.parameters()) + 
            list(self.classifier.parameters()), 
            lr=args.lr, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=1e-3,
        )
        # 使用 DataParallel 在多 GPU 上并行训练
        self.ecg_model = nn.DataParallel(self.ecg_model).to(self.device)
        self.cxr_model = nn.DataParallel(self.cxr_model).to(self.device)
        self.classifier = nn.DataParallel(self.classifier).to(self.device)
        # self.scheduler = CosineAnnealingLR(self.optimizer, self.args.epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3, mode='min')
        projection_dim = 256


# 添加线性投影层
        # if self.args.domain=='S_T':
            # self.ecg_projection = nn.Linear(512, projection_dim)
        self.ecg_projection = nn.Linear(512, projection_dim)  # 投影 ecg_feats
        if self.args.domain=='S_T':
            self.cxr_projection = nn.Linear(512, projection_dim)
        else:
            self.cxr_projection = nn.Linear(448, projection_dim)
        self.ecg_projection = self.ecg_projection.to(self.device)
        self.cxr_projection = self.cxr_projection.to(self.device)  # 投影 cxr_feats

#TODO: to mark here ues the pretrained model
        if self.args.fusion_type=='fusion' and self.args.domain=='frequency':
            print(f'cxr_frequency')
            self.load_pretrained_weights()
#TODO: to mark here ues the pretrained model
        self.best_auroc = 0
        self.best_stats = None
        # self.epochs_stats = {'epoch':[],'loss train': [], 'auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
        #                      'loss val': [],'auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}
        # self.epochs_stats = {'epoch':[],'loss train': [],'f1 train' :[],'macro_auroc train': [], 'micro_auroc train': [], 'weighted_auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
        #                                 'loss val': [],'f1 val' :[],'macro_auroc val': [],'micro_auroc val': [], 'weighted_auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}
        self.epochs_stats = {'epoch':[],'loss train': [],'f1 train' :[],'macro_auroc train': [], 'micro_auroc train': [], 'weighted_auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
                                        'loss val': [],'f1 val' :[],'macro_auroc val': [],'micro_auroc val': [], 'weighted_auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}

    def load_pretrained_weights(self):
        # 加载预训练权重并替换最后一层
        checkpoint = torch.load('/home/mimic/MIMIC_subset/MIMIC_subset/wavevit_s.pth.tar', weights_only=True,map_location=self.device)
        
        # 删除不需要的头部权重
        del checkpoint['state_dict']['head.weight']
        del checkpoint['state_dict']['head.bias']
        del checkpoint['state_dict']['aux_head.weight']
        del checkpoint['state_dict']['aux_head.bias']
        
        # 加载权重
        self.cxr_model.load_state_dict(checkpoint['state_dict'], strict=False)
        # self.model.to(self.device)

        # 替换最后一层以适应新任务
        num_classes = 7  # 根据你的数据集类别数修改
        self.cxr_model.module.head = nn.Linear(self.cxr_model.module.head.in_features, num_classes)
        self.cxr_model.to(self.device)



    # def train(self):
    #     self.ecg_model.train()
    #     self.cxr_model.train()
    def train_epoch(self):
        print(f'==================starting train epoch {self.epoch}==========================')
        print('Current learning rate: ',self.optimizer.param_groups[0]['lr'])
        epoch_loss=0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        
        for batch_idx, (ecg_data, cxr_data, target) in enumerate(self.train_dl):
            ecg_data_np = np.array(ecg_data)


            ecg_data = torch.from_numpy(ecg_data_np).float()
            ecg_data = ecg_data.float().to(self.device)
            cxr_data = cxr_data.to(self.device)
            target = torch.from_numpy(target).float()
            target = target.to(self.device)

            # 获取 ECG 特征
            ecg_feats, _ = self.ecg_model(ecg_data)
            # print(f'ecg_feats {ecg_feats.shape}')  # [batch,512]
            ecg_feats = self.ecg_projection(ecg_feats)  # 将 ecg_feats 投影到 256 维
            # print(f'Projected ecg_feats {ecg_feats.shape}')  # [batch,256]

            # 获取 CXR 特征
            cxr_feats, _ = self.cxr_model(cxr_data)
            cxr_feats = cxr_feats.view(cxr_feats.size(0), -1)
            cxr_feats = self.cxr_projection(cxr_feats)  # 将 cxr_feats 投影到 256 维
            # print(f'Projected cxr_feats {cxr_feats.shape}')  # [batch,256]

            # print(f'cxr_feats {cxr_feats.shape}')  # [batch,448]

            # 特征拼接
            fused_feats = torch.cat((ecg_feats, cxr_feats), dim=1)
            # print(f'fused_feats {fused_feats.shape}')  # [batch,960]

            # 通过分类器进行预测
            outputs = self.classifier(fused_feats)
            

            # 计算损失
            loss = self.criterion(outputs, target)
            epoch_loss+=loss.item()

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED,outputs),0)
            outGT = torch.cat((outGT,target),0)

        # print(f'Epoch [{epoch+1}/{self.args.epochs}], Loss: {loss.item():.4f}')
        print(f"lr {self.args.lr} train [{self.epoch:04d} / {self.args.epochs:04d}] train loss: \t{epoch_loss/batch_idx:0.5f} ")
        outPRED_sigmoid = torch.sigmoid(outPRED)
        ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')#之前outPRED_sigmoid处为outPRED
        # self.scheduler.step()
        micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')

# 计算加权 AUROC
        weighted_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='weighted')
        # self.scheduler.step()


        print(f'train macro AUC {ret}')
        print(f'train micro AUC {micro_auc}')
        print(f'train weighted AUC {weighted_auc}')

        # print(f'train AUC {ret}')
        pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

        # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())#之前outPRED_sigmoid处为outPRED
        # pr_auc = auc(recall, precision)
        print(f'train PR AUC: {pr_auc}')
        y_true = outGT.data.cpu().numpy().ravel()  # 真实标签
        y_pred_prob = outPRED_sigmoid.data.cpu().numpy().ravel()  # Sigmoid 后的预测概率

        # 将预测概率转换为二值预测 (0 或 1)，阈值通常为 0.5
        y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

        # 计算 F1 Score
        f1 = f1_score(y_true, y_pred)
        self.epochs_stats['f1 train'].append(f1)
        print("F1 Score:", f1)
        self.epochs_stats['loss train'].append(epoch_loss/batch_idx)
        self.epochs_stats['micro_auroc train'].append(micro_auc)
        self.epochs_stats['weighted_auroc train'].append(weighted_auc)

        # 'micro_auroc train': [], 'weighted_auroc train': []
        self.epochs_stats['macro_auroc train'].append(ret)

        # self.epochs_stats['loss train'].append(epoch_loss/batch_idx)
        # self.epochs_stats['auroc train'].append(ret)
        self.epochs_stats['auprc train'].append(pr_auc)
        
        for i in range(outGT.shape[1]):  # 假设有7个类别，outGT 是 [batch_size, num_classes]
            class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
            # print(f'train AUC for class {i + 1}: {class_auroc}')
            
            # 根据类别号将 AUROC 追加到对应的 key
            self.epochs_stats[f'auroc {i + 1} train'].append(class_auroc)

        # print("Training completed.")

    def validate(self,dl):
        print(f'-----------------starting val epoch {self.epoch}--------------------')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        # self.ecg_model.eval()
        # self.cxr_model.eval()
        # total_loss = 0.0
        # correct = 0
        # total = 0
        
        with torch.no_grad():
            for batch_idx, (ecg_data, cxr_data, target) in enumerate(dl):
                # print(f'ecg_data {ecg_data}')
                # print(f'cxr_data {cxr_data}')
                # print(f'target {target}')
                ecg_data_np = np.array(ecg_data)


                ecg_data = torch.from_numpy(ecg_data_np).float()
                ecg_data = ecg_data.float()
                # cxr_data = cxr_data.to(self.device)
                target = torch.from_numpy(target).float()

                ecg_data= Variable(ecg_data.to(self.device),requires_grad=False)
                cxr_data= Variable(cxr_data.to(self.device),requires_grad=False)
                
                target= Variable(target.to(self.device),requires_grad=False)


                # 获取 ECG 特征
                ecg_feats,_ = self.ecg_model(ecg_data)
                # print(f'1 ecg {ecg_feats.shape}')
                ecg_feats = self.ecg_projection(ecg_feats)  # 将 ecg_feats 投影到 256 维
                # print(f'2 ecg {ecg_feats.shape}')
                ecg_feats.to(self.device)
                # 获取 CXR 特征
                cxr_feats,_ = self.cxr_model(cxr_data)
                # print(f'1 cxr {cxr_feats.shape}')
                cxr_feats = cxr_feats.view(cxr_feats.size(0), -1)
                # print(f'2 cxr {cxr_feats.shape}')
                cxr_feats = self.cxr_projection(cxr_feats)  # 将 cxr_feats 投影到 256 维
                # print(f'3 cxr {cxr_feats.shape}')
                cxr_feats.to(self.device)

                # 特征拼接
                fused_feats = torch.cat((ecg_feats, cxr_feats), dim=1)

                # 通过分类器进行预测
                outputs = self.classifier(fused_feats)
                # print(f'val outputs {outputs.shape}')
                # 计算损失
                loss = self.criterion(outputs, target)
                # print(f'val loss {loss}')
                # total_loss += loss.item()
                epoch_loss+=loss.item()
                # print(f' each val epoch_loss {epoch_loss}')
                outPRED = torch.cat((outPRED, outputs), 0)
                outGT = torch.cat((outGT, target), 0)

                # # 计算准确率
                # _, predicted = torch.max(outputs.data, 1)
                # total += target.size(0)
                # correct += (predicted == target).sum().item()
                
            self.scheduler.step(epoch_loss/len(self.val_dl))

            print(f'total val epoch loss {epoch_loss}')
            print(f'batch_idx {batch_idx}')
            print(f"lr {self.args.lr} val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/batch_idx:0.5f} ")
            outPRED_sigmoid = torch.sigmoid(outPRED)
            ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')#之前outPRED_sigmoid处为outPRED
            # print(f'val AUC {ret}')
            micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')

            weighted_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='weighted')
            print(f'val micro AUC {micro_auc}')
            print(f'val weighted AUC {weighted_auc}')

            pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

            print(f'val PR AUC: {pr_auc}')


            # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())#之前outPRED_sigmoid处为outPRED
            # pr_auc = auc(recall, precision)
            pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

            print(f'val PR AUC: {pr_auc}')
            # self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            y_true = outGT.data.cpu().numpy().ravel()  # 真实标签
            y_pred_prob = outPRED_sigmoid.data.cpu().numpy().ravel()  # Sigmoid 后的预测概率

            # 将预测概率转换为二值预测 (0 或 1)，阈值通常为 0.5
            y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

            # 计算 F1 Score
            f1 = f1_score(y_true, y_pred)
            self.epochs_stats['f1 val'].append(f1)
            print("F1 Score:", f1)

            self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            self.epochs_stats['macro_auroc val'].append(ret)
            self.epochs_stats['weighted_auroc val'].append(weighted_auc)

        # 'micro_auroc train': [], 'weighted_auroc train': []
            self.epochs_stats['micro_auroc val'].append(micro_auc)

            self.epochs_stats['auprc val'].append(pr_auc)
            for i in range(outGT.shape[1]):  # 假设有7个类别，outGT 是 [batch_size, num_classes]
                class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
                print(f'train AUC for class {i + 1}: {class_auroc}')
                
                # 根据类别号将 AUROC 追加到对应的 key
                self.epochs_stats[f'auroc {i + 1} val'].append(class_auroc)

            return ret

    def save_epochs_stats(self, file_path='result.txt'):
    # 将字典写入文件 result.txt
        with open(file_path, 'w') as f:
            json.dump(self.epochs_stats, f, indent=4)

        # print(f'Validation Loss: {total_loss/len(self.val_dl):.4f}, Accuracy: {100 * correct / total:.2f}%')
    def train(self):
        #TODO:change epoch number
        for self.epoch in range(0, self.args.epochs):
            print(f'patience: {self.patience}')
            # self.model.eval()
            self.epochs_stats['epoch'].append(self.epoch)
            self.ecg_model.eval()
            self.cxr_model.eval()
            ret=self.validate(self.val_dl)
#TODO:还没有save——checkpoint           
            # self.save_checkpoint(prefix='last')

            if self.best_auroc < ret:
                self.best_auroc = ret
                self.best_stats = ret
#TODO:还没有save——checkpoint 
                # self.save_checkpoint()
                # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                # self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                # self.print_and_write(ret, isbest=False)
                
                self.patience+=1
            




            # self.model.train()
            self.ecg_model.train()
            self.cxr_model.train()
            self.train_epoch()
            if self.patience >= self.args.patience:
                break

            if args.domain=='frequency':
                file_path = f'result_record/G1_{args.domain}_fusion_result.txt'
                self.save_epochs_stats(file_path)
            elif args.domain=='S_T':
                file_path = f'result_record/G1_{args.domain}_fusion_result.txt'
                self.save_epochs_stats(file_path)

            # self.save_epochs_stats('frequency_fusion_trainer_result.txt')

    # def train(self):
    #     # 设置为训练模式
    #     self.ecg_model.train()
    #     self.cxr_model.train()
        
    #     # TODO: 迭代每个 epoch
    #     for self.epoch in range(self.args.epochs):
    #         print(f"Epoch [{self.epoch + 1}/{self.args.epochs}]")
            
    #         # 训练过程
    #         self.train_epoch()  # 调用训练单个 epoch 的函数
            
    #         # 验证过程
    #         self.ecg_model.eval()
    #         self.cxr_model.eval()
    #         val_result = self.validate(self.val_dl)  # 验证函数需要实现
            
    #         # 根据验证结果更新最佳 AUROC
    #         if self.best_auroc < val_result['auroc_mean']:
    #             self.best_auroc = val_result['auroc_mean']
    #             self.save_checkpoint()  # 保存当前最佳模型
            
    #         print(f'Best AUROC: {self.best_auroc:.4f}')

    #         # 保存最后一次的模型
    #         self.save_checkpoint(prefix='last')


# 实例化 Fusion_trainer
# trainer = Fusion_trainer(train_dl, val_dl, args, spectrogram_model(), wavevit_s())

# # 训练模型
# trainer.train()

# # 验证模型
# trainer.validate()



