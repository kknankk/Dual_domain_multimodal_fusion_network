
import os
import sys
import json
sys.path.append(os.path.abspath('/data/mimic/MIMIC_subset/MIMIC_subset'))
import torch
from train.trainer_utils import Trainer
from dataset.update_ECGdataset import get_ECG_datasets,get_data_loader
import numpy as np
# from model.ECG_model import LSTM,Spect_CNN,ECGModel
from model.fusion_model import FSRU
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from torch.autograd import Variable
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
torch.set_printoptions(profile="full")#使得完全打印
torch.set_printoptions(threshold=10000)  # 可以根据数据大小调整阈值
torch.set_printoptions(threshold=torch.inf, linewidth=1000)  # threshold为inf，linewidth设置为较大值
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc,average_precision_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
from dataset.fusion_dataset import get_ecgcxr_data_loader
train_dl, val_dl = get_ecgcxr_data_loader(batch_size=args.batch_size)
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #use first four GPU


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        masks=p.size(0)
        # print(f'mask {masks}')
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks)



class fourinput_saj_trainer(Trainer):
    def __init__(self,
        train_dl,
        val_dl,
        args,
        model,
        test_dl=None):
#TODO: 目前只使用一个，本来batch size是16，指定4个gpu每个batch size就为4；指定3个gpu batchsize就为6，4，6        
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #use first four GPU

        super(fourinput_saj_trainer,self).__init__(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #use LSTM to verify first
        # self.model=LSTM(input_dim=12, num_classes=7,  dropout=0.2, layers=2)
        # self.model=Spect_CNN()
        # self.model=ECGModel()
        self.model=model
        # device_ids = [0, 1]
        # self.device = 'cuda:{}'.format(device_ids[0])
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        # self.model=model
        # self.model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        # self.model = self.model.to(self.device)
        self.jsd = JSD()
        self.epoch=0
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # self.loss = nn.BCELoss()
        # self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.7,4.1,2.87,1.69,4.56,5.58,10.36]).to(self.device))#each=N/P
        # self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.7,3.1,1.9,0.7,3.6,4.6,9.4]).to(self.device)) 
        # self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.6,3.4,1.8,0.68,3.6,4.3,9.5]).to(self.device)) 
        # self.loss=nn.BCEWithLogitsLoss() 
        self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.63,1.52,1.36,0.24]).to(self.device)) 

        self.optimizer=optim.Adam(self.model.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay,)#之前是1e-3
        # Resnet34 for CXR : lr=0.0001
    #TODO write the load_state func
        # self.scheduler=ReduceLROnPlateau(self.optimizer,factor=0.5, patience=10, mode='min')

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=2, mode='min')
        # self.scheduler = CosineAnnealingLR(self.optimizer, self.args.epochs, eta_min=1e-6, last_epoch=-1)

#TODO: to mark here ues the pretrained model
        if self.args.fusion_type=='cxr' and self.args.domain=='frequency':
            print(f'cxr_frequency')
            self.load_pretrained_weights()
#TODO: to mark here ues the pretrained model


        self.best_auroc = 0
        self.best_stats = None
        # self.epochs_stats = {'epoch':[],'loss train': [], 'auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
        #                      'loss val': [],'auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}
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
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # self.model.to(self.device)

        # 替换最后一层以适应新任务
        num_classes = 4  # 根据你的数据集类别数修改
        self.model.module.head = nn.Linear(self.model.module.head.in_features, num_classes)
        self.model.to(self.device)



    def train_epoch(self):
        print(f'==================starting train epoch {self.epoch}==========================')
        print('Current learning rate: ',self.optimizer.param_groups[0]['lr'])
        epoch_loss=0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        
        
        for batch_idx,(ecg_data,cxr_data,target,_,_) in enumerate(self.train_dl):
            # print(f'deeper_trainer ecg 0 {ecg_data[1].dtype}') #[B,257,17,12],complex
            # if self.args.fusion_type=='deeper_frequency_fusion':
            #     ecg_data_np = np.array(ecg_data)
            #     ecg_data = torch.from_numpy(ecg_data_np).to(torch.complex64).to(self.device)
            # else:
            # print(f'batch_idx {batch_idx}')
            ecg_data_np = np.array(ecg_data)
            ecg_data = torch.from_numpy(ecg_data_np).float()
            ecg_data = ecg_data.float().to(self.device)
            cxr_data = cxr_data.to(self.device)
            target = torch.from_numpy(target).float()
            target = target.to(self.device)
            # print(f'deeper_trainer ecg {ecg_data.dtype}') #[B,257,17,12]complex
            # print(f'deeper_trainer 1 ecg {ecg_data.shape}')
            # print(f'deeper_trainer cxr {cxr_data.shape} ')#[B,3,224,224]

            # _,output = self.model(data)
            # print(f'model {self.model}')
            text_outputs, image_outputs, output, _ = self.model(ecg_data, cxr_data)
            # print(f'train_epoch batch output {output}')
            # output_prob = F.softmax(output,dim=1)
            # print(f'general trainer output {output.shape}')
            # print(f'text_outputs {text_outputs.shape}')
            jsd_loss=self.jsd(text_outputs.sigmoid(),image_outputs.sigmoid())
            loss = self.loss(output,target)
            loss=loss+jsd_loss
#======add flooding=====
            # loss = (loss - self.args.b).abs() + self.args.b
#======add flooding=====
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
        print(f"lr {self.args.lr} train [{self.epoch:04d} / {self.args.epochs:04d}] train loss: \t{epoch_loss/batch_idx:0.5f} ")
        # print(f'train epoch output {outPRED}')
        outPRED_sigmoid = torch.sigmoid(outPRED)
        # print(f'outPRED_sigmoid {outPRED_sigmoid.shape}')
        ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')#之前outPRED_sigmoid处为outPRED
        # self.scheduler.step()
        micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')

# 计算加权 AUROC
        weighted_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='weighted')
        # self.scheduler.step()


        print(f'train macro AUC {ret}')
        print(f'train micro AUC {micro_auc}')
        print(f'train weighted AUC {weighted_auc}')

        # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())#之前outPRED_sigmoid处为outPRED
        # pr_auc = auc(recall, precision)
        pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

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
        self.epochs_stats['macro_auroc train'].append(ret)
        self.epochs_stats['weighted_auroc train'].append(weighted_auc)

        # 'micro_auroc train': [], 'weighted_auroc train': []
        self.epochs_stats['micro_auroc train'].append(micro_auc)

        
        # self.epochs_stats['loss train'].append(epoch_loss/batch_idx)
        # self.epochs_stats['auroc train'].append(ret)
        self.epochs_stats['auprc train'].append(pr_auc)

        for i in range(outGT.shape[1]):  # 假设有7个类别，outGT 是 [batch_size, num_classes]
            class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
            # print(f'train AUC for class {i + 1}: {class_auroc}')
            
            # 根据类别号将 AUROC 追加到对应的 key
            self.epochs_stats[f'auroc {i + 1} train'].append(class_auroc)




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
            for batch_idx, (ecg_data, cxr_data, target,_,_) in enumerate(dl):
                # print(f'ecg_data {ecg_data}')
                # print(f'cxr_data {cxr_data}')
                # print(f'target {target}')
                # if self.args.fusion_type=='deeper_frequency_fusion':
                #     ecg_data_np = np.array(ecg_data)
                #     ecg_data = torch.from_numpy(ecg_data_np).to(torch.complex64).to(self.device)
                # else:

                ecg_data_np = np.array(ecg_data)


                ecg_data = torch.from_numpy(ecg_data_np).float()
                ecg_data = ecg_data.float()
                # cxr_data = cxr_data.to(self.device)
                target = torch.from_numpy(target).float()

                ecg_data= Variable(ecg_data.to(self.device),requires_grad=False)
                cxr_data= Variable(cxr_data.to(self.device),requires_grad=False)
               
                target= Variable(target.to(self.device),requires_grad=False)

# #another eval:
#                 all_preds.append(output)
#                 all_labels.append(target)
# #another eval:
                # print(f'val output {index1} {output}')
                # output_prob=F.softmax(output,dim=1)
                text_outputs, image_outputs, output, _ = self.model(ecg_data, cxr_data)

                loss=self.loss(output,target)
                # print(f'each val loss{loss}')
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
            print(f"lr {self.args.lr} val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/batch_idx:0.5f} ")
            # ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            outPRED_sigmoid = torch.sigmoid(outPRED)
            ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')#之前outPRED_sigmoid处为outPRED
            print(f'val AUC {ret}')
            # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())#之前outPRED_sigmoid处为outPRED
            # pr_auc = auc(recall, precision)
            micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')

# 计算加权 AUROC
            weighted_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='weighted')
            print(f'val micro AUC {micro_auc}')
            print(f'val weighted AUC {weighted_auc}')

            pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

            print(f'val PR AUC: {pr_auc}')
            # np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
            # np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 

            # self.epochs_stats['auroc val'].append(ret['auroc_mean'])
            # self.epochs_stats['auprc val'].append(ret['auprc_mean'])

            # self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # print(f'epoch_state {self.epochs_stats}')
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

            # self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            self.epochs_stats['auprc val'].append(pr_auc)
            for i in range(outGT.shape[1]):  # 假设有7个类别，outGT 是 [batch_size, num_classes]
                class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
                # print(f'train AUC for class {i + 1}: {class_auroc}')
                
                # 根据类别号将 AUROC 追加到对应的 key
                self.epochs_stats[f'auroc {i + 1} val'].append(class_auroc)

            return ret

    def save_epochs_stats(self, file_path='result.txt'):
    # 将字典写入文件 result.txt
        with open(file_path, 'w') as f:
            json.dump(self.epochs_stats, f, indent=4)

#TODO: add test func

    def train(self):
        #TODO:change epoch number
        for self.epoch in range(self.start_epoch, self.args.epochs):
            # self.model.eval()
            print(f'patience: {self.patience}')
            self.model.train()
            self.train_epoch()
            self.epochs_stats['epoch'].append(self.epoch)
            # ret=self.validate(self.val_dl)
            self.save_checkpoint(prefix='last')
            # print(f'self.best_auroc {self.best_auroc}')
            # a=ret['auroc_mean']
            # print(f'auroc_mean {a}')
            # if self.best_auroc < ret['auroc_mean']:
            #     self.best_auroc = ret['auroc_mean']
            self.model.eval()
            ret=self.validate(self.val_dl)
            if self.best_auroc < ret:
                self.best_auroc = ret
                self.save_checkpoint()
                self.patience = 0
            else:
                # self.print_and_write(ret, isbest=False)
                self.patience+=1
            print(f'self.best_auroc {self.best_auroc}')
            if self.patience >= self.args.patience:
                print(f'{self.patience}>{self.args.patience}')
                break
            # self.model.train()
            # self.train_epoch()
            # self.model.eval()
            # ret=self.validate(self.val_dl)
            # self.save_epochs_stats('G_trainer_result.txt')
            # self.save_epochs_stats(f'{G_self.args.module_self.args.model_result}.txt')
            if args.fusion_type=='deeper_frequency_fusion':
                file_path = f'result_record/G3_{args.name}_{args.fusion_type}_{args.fusion_model}_result.txt'
                self.save_epochs_stats(file_path)
            elif args.fusion_type=='fourinput_saj_fusion':
                file_path = f'result_record/G3_{args.name}_{args.fusion_type}_{args.fusion_model}_result.txt'



            
    



            
            