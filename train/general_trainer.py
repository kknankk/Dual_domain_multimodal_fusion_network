
import os
import sys
sys.path.append(os.path.abspath('/data/ke/MIMIC_subset'))
import torch
from train.trainer_utils import Trainer
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
import numpy as np
from model.ECG_model import LSTM,Spect_CNN,ECGModel
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
# torch.set_printoptions(profile="full")#使得完全打印
# torch.set_printoptions(threshold=10000)  # 可以根据数据大小调整阈值
# torch.set_printoptions(threshold=torch.inf, linewidth=1000)  # threshold为inf，linewidth设置为较大值
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
import torch.nn.functional as F
from sklearn.metrics import f1_score

class G_trainer(Trainer):
    def __init__(self,
        train_dl,
        val_dl,
        args,
        model,
        test_dl=None):
#TODO: 目前只使用一个，本来batch size是16，指定4个gpu每个batch size就为4；指定3个gpu batchsize就为6，4，6        
        os.environ["CUDA_VISIBLE_DEVICES"] = "6" #use first four GPU

        super(G_trainer,self).__init__(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #use LSTM to verify first
        # self.model=LSTM(input_dim=12, num_classes=7,  dropout=0.2, layers=2)
        # self.model=Spect_CNN()
        # self.model=ECGModel()
        # self.model = torch.nn.DataParallel(self.model)
        self.model = model.to(self.device)

        self.epoch=0
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # self.loss = nn.BCELoss()
        # self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.7,4.1,2.87,1.69,4.56,5.58,10.36]).to(self.device))
        self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8,4,3,2,5,6,10.4]).to(self.device))

        self.optimizer=optim.Adam(self.model.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-3)
    #TODO write the load_state func
        # self.scheduler=ReduceLROnPlateau(self.optimizer,factor=0.5, patience=10, mode='min')
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc train': [],'auroc val': [],'auprc train':[],'auprc val':[]}


    def train_epoch(self):
        print(f'==================starting train epoch {self.epoch}==========================')
        epoch_loss=0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        
        for batch_idx,(data,target) in enumerate(self.train_dl):
            # print(f'trainer ecg_data:{ecg_data}')
            data=torch.from_numpy(data).float()
            data=data.to(self.device)
            target = torch.from_numpy(target).float()
            target=target.to(self.device)

            output = self.model(data)
            # print(f'train_epoch batch output {output}')
            # output_prob = F.softmax(output,dim=1)
            # print(f'general trainer output {output}')
            loss = self.loss(output,target)
            # print(f'batch_loss {loss}')
            epoch_loss+=loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # outPRED = torch.cat((outPRED,output_prob),0)
            outPRED = torch.cat((outPRED,output),0)
            # print(f'outPRED shape {outPRED.shape}')
            outGT = torch.cat((outGT,target),0)
            # print(f'outGT shape {outGT.shape}')

            # if batch_idx%20 ==0:
            #TODO: 删除下面两行的#
            # if batch_idx != 0:
            #     print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{batch_idx:04}/{steps}]   lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/batch_idx:0.5f} ")

            # print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{batch_idx:04}/{steps}]   lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/batch_idx:0.5f} ")
            
        # ret = self.computeAUROC(outGT.data.to(self.device).cpu().numpy(), outPRED.data.to(self.device).cpu().numpy(), 'train')
        print(f"train [{self.epoch:04d} / {self.args.epochs:04d}] train loss: \t{epoch_loss/batch_idx:0.5f} ")

        ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), average='macro')
        
        # self.epochs_stats['auroc train'].append(ret['auroc_mean'])
        # self.epochs_stats['auprc train'].append(ret['auprc_mean'])
        
        # self.epochs_stats['loss train'].append(epoch_loss/batch_idx)
        # print(f'epoch_state {self.epochs_stats}')
        print(f'train AUC {ret}')
        precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED.data.cpu().numpy().ravel())
        pr_auc = auc(recall, precision)
        print(f'train PR AUC: {pr_auc}')


#TODO: add validate func
    def validate(self,dl):
        print(f'-----------------starting val epoch {self.epoch}--------------------')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        all_preds=[]
        all_labels=[]

        with torch.no_grad():
            for batch_idx,(data,target) in enumerate(dl):
                target=self.get_gt(target)
                data=torch.from_numpy(data).float()
                data= Variable(data.to(self.device),requires_grad=False)
                target= Variable(target.to(self.device),requires_grad=False)
                # print(f'whole ecg {ecg_data}')
                # print(f'4 th ecg 8th col: {ecg_data[3][:,7]}')#12leads的第8lead是nan
                
                output=self.model(data)
# #another eval:
#                 all_preds.append(output)
#                 all_labels.append(target)
# #another eval:
                # print(f'output {output}')
                # output_prob=F.softmax(output,dim=1)
                loss=self.loss(output,target)
                # print(f'each loss{loss}')
                epoch_loss+=loss.item()
                # print(f'epoch loss{epoch_loss}')
                outPRED = torch.cat((outPRED, output), 0)
                outGT = torch.cat((outGT, target), 0)
# #another eval:
#             all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
#             all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
#             threshold = 0.5
#             predicted_labels = (all_preds > threshold).astype(int)
#             f1 = f1_score(all_labels, predicted_labels, average='weighted')
#             print(f'Test F1 Score: {f1}')


# # another eval:        

            # print(f'val batch_idx {batch_idx}')
            self.scheduler.step(epoch_loss/len(self.val_dl))
            #TODO:z下一行delete # 
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/batch_idx:0.5f} ")
            # ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), average='macro')
            print(f'val AUC {ret}')
            precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED.data.cpu().numpy().ravel())
            pr_auc = auc(recall, precision)
            print(f'val PR AUC: {pr_auc}')
            # np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
            # np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 

            # self.epochs_stats['auroc val'].append(ret['auroc_mean'])
            # self.epochs_stats['auprc val'].append(ret['auprc_mean'])

            # self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # print(f'epoch_state {self.epochs_stats}')
            return ret



#TODO: add test func

    def train(self):
        #TODO:change epoch number
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret=self.validate(self.val_dl)
            self.save_checkpoint(prefix='last')
            # print(f'self.best_auroc {self.best_auroc}')
            # a=ret['auroc_mean']
            # print(f'auroc_mean {a}')
            # if self.best_auroc < ret['auroc_mean']:
            #     self.best_auroc = ret['auroc_mean']
            if self.best_auroc < ret:
                self.best_auroc = ret
                self.save_checkpoint()
            print(f'self.best_auroc {self.best_auroc}')

            self.model.train()
            self.train_epoch()

            
    



            
            
