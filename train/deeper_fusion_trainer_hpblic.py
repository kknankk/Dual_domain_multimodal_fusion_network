
import os
import sys
import json
sys.path.append(os.path.abspath('/data/mimic/MIMIC_subset/MIMIC_subset'))
import torch
from train.trainer_utils import Trainer
from dataset.update_ECGdataset import get_ECG_datasets,get_data_loader
import numpy as np
# from model.ECG_model import LSTM,Spect_CNN,ECGModel
# from model.fusion_model import FSRU
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from torch.autograd import Variable
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc,average_precision_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
from dataset.fusion_dataset_ONE import get_ecgcxr_data_loader
train_dl, val_dl = get_ecgcxr_data_loader(batch_size=args.batch_size)
from sklearn.metrics import f1_score
import torch
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" 
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12365'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

class deeper_fusion_trainer_hpblic(Trainer):
    def __init__(self,
        train_dl,
        val_dl,
        args,
        model,
        test_dl=None):
     
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        super(deeper_fusion_trainer_hpblic,self).__init__(args)
        dist.init_process_group(backend='nccl')

        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model=model

        self.model = self.model.to(self.device)

        self.model = torch.nn.DataParallel(self.model)


        self.epoch=0
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))


        self.loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.88,2.34,2.37]).to(self.device)) 


        self.optimizer=optim.Adam(self.model.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay,)#之前是1e-3

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=2, mode='min')

        if self.args.fusion_type=='cxr' and self.args.domain=='frequency':
            print(f'cxr_frequency')
            self.load_pretrained_weights()



        self.best_auroc = 0
        self.best_stats = None
        # self.epochs_stats = {'epoch':[],'loss train': [], 'auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
        #                      'loss val': [],'auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}
        self.epochs_stats = {'epoch':[],'loss train': [],'f1 train' :[],'macro_auroc train': [], 'micro_auroc train': [], 'weighted_auroc train': [],'auroc 1 train': [],'auroc 2 train': [],'auroc 3 train': [],'auroc 4 train': [],'auroc 5 train': [],'auroc 6 train': [],'auroc 7 train': [],'auprc train':[],
                                        'loss val': [],'f1 val' :[],'macro_auroc val': [],'micro_auroc val': [], 'weighted_auroc val': [],'auroc 1 val': [],'auroc 2 val': [],'auroc 3 val': [],'auroc 4 val': [],'auroc 5 val': [],'auroc 6 val': [],'auroc 7 val': [],'auprc val':[]}
        self.best_checkpoint_path = f'/home/mimic/MIMIC_subset/MIMIC_subset/checkpoints/deeper_frequency_fusion_mod/all_addcli1216/frequency/best_checkpoint.pth.tar'

    def load_best_checkpoint(self):
        # Load the model state from the best checkpoint
        checkpoint = torch.load(self.best_checkpoint_path)
        # print(f"Checkpoint keys: {checkpoint.keys()}")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()  # Set the model to evaluation mode

    def load_pretrained_weights(self):

        checkpoint = torch.load('/home/mimic/MIMIC_subset/MIMIC_subset/wavevit_s.pth.tar', weights_only=True,map_location=self.device)
        

        del checkpoint['state_dict']['head.weight']
        del checkpoint['state_dict']['head.bias']
        del checkpoint['state_dict']['aux_head.weight']
        del checkpoint['state_dict']['aux_head.bias']
        

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)


        self.model.module.head = nn.Linear(self.model.module.head.in_features, num_classes)
        self.model.to(self.device)



    def train_epoch(self):
        print(f'==================starting train epoch {self.epoch}==========================')
        print('Current learning rate: ',self.optimizer.param_groups[0]['lr'])
        each_loss=0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        
        
        for batch_idx,(ecg_data,cxr_data,clic_value,target,_,_) in enumerate(self.train_dl):

            ecg_data_np = np.array(ecg_data)
            ecg_data = torch.from_numpy(ecg_data_np).float()
            ecg_data = ecg_data.float().to(self.device)
            cxr_data = cxr_data.to(self.device)
            target = torch.from_numpy(target).float()
            target = target.to(self.device)

            text_outputs, image_outputs, output, total_loss = self.model(ecg_data, cxr_data,clic_value)

            loss = 0.8*self.loss(output,target)

            total_loss=total_loss.mean()

            loss+=0.2*total_loss.item()
            each_loss+=loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # outPRED = torch.cat((outPRED,output_prob),0)
            outPRED = torch.cat((outPRED,output),0)
            # print(f'outPRED shape {outPRED.shape}')
            outGT = torch.cat((outGT,target),0)
            # print(f'outGT shape {outGT.shape}')

            # if batch_idx%20 ==0:



        print(f"lr {self.args.lr} train [{self.epoch:04d} / {self.args.epochs:04d}] train loss: \t{each_loss/batch_idx:0.5f} ")

        outPRED_sigmoid = torch.sigmoid(outPRED)
        # outPRED_sigmoid=outPRED
        # print(f'outPRED_sigmoid {outPRED_sigmoid.shape}')
        ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')#之前outPRED_sigmoid处为outPRED
        # self.scheduler.step()
        micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')


        weighted_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='weighted')
        # self.scheduler.step()


        print(f'train macro AUC {ret}')
        print(f'train micro AUC {micro_auc}')
        print(f'train weighted AUC {weighted_auc}')

        # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())
        # pr_auc = auc(recall, precision)
        pr_auc=average_precision_score(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())

        print(f'train PR AUC: {pr_auc}')
        y_true = outGT.data.cpu().numpy().ravel()  
        y_pred_prob = outPRED_sigmoid.data.cpu().numpy().ravel()  

        y_pred = np.where(y_pred_prob >= 0.5, 1, 0)


        f1 = f1_score(y_true, y_pred)
        self.epochs_stats['f1 train'].append(f1)
        print("F1 Score:", f1)
        self.epochs_stats['loss train'].append(loss/batch_idx)
        self.epochs_stats['macro_auroc train'].append(ret)
        self.epochs_stats['weighted_auroc train'].append(weighted_auc)

        # 'micro_auroc train': [], 'weighted_auroc train': []
        self.epochs_stats['micro_auroc train'].append(micro_auc)

        
        # self.epochs_stats['loss train'].append(epoch_loss/batch_idx)
        # self.epochs_stats['auroc train'].append(ret)
        self.epochs_stats['auprc train'].append(pr_auc)

        for i in range(outGT.shape[1]):  
            class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())

            self.epochs_stats[f'auroc {i + 1} train'].append(class_auroc)





    def validate(self,dl):
        print(f'-----------------starting val epoch {self.epoch}--------------------')
        each_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        all_preds=[]
        all_labels=[]

        with torch.no_grad():
            for batch_idx, (ecg_data, cxr_data,clic_value, target,_,_) in enumerate(dl):
                

                ecg_data_np = np.array(ecg_data)


                ecg_data = torch.from_numpy(ecg_data_np).float()
                ecg_data = ecg_data.float()
                # cxr_data = cxr_data.to(self.device)
                target = torch.from_numpy(target).float()

                ecg_data= Variable(ecg_data.to(self.device),requires_grad=False)
                cxr_data= Variable(cxr_data.to(self.device),requires_grad=False)
               
                target= Variable(target.to(self.device),requires_grad=False)


                text_outputs, image_outputs, output, total_loss = self.model(ecg_data, cxr_data,clic_value)



                loss = 0.8*self.loss(output,target)
            # print
                total_loss=total_loss.mean()
            # print(f'loss {loss}')
            # print(f'total loss {total_loss}')
                loss+=0.2*total_loss.item()
                each_loss+=loss.item()

                # loss=self.loss(output,target)
                # # print(f'each val loss{loss}')
                # epoch_loss+=loss.item()
                # total_loss=total_loss.mean()
                # epoch_loss+=total_loss.item()
                # print(f'epoch loss{epoch_loss}')
                outPRED = torch.cat((outPRED, output), 0)
                outGT = torch.cat((outGT, target), 0)

            print(f"lr {self.args.lr} val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{each_loss/batch_idx:0.5f} ")
            # ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            outPRED_sigmoid = torch.sigmoid(outPRED)
            # outPRED_sigmoid=outPRED
            ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')
            print(f'val AUC {ret}')
            # self.scheduler.step(each_loss/len(self.val_dl))
            # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())
            self.scheduler.step(ret)
            # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())
            # pr_auc = auc(recall, precision)
            micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')


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
            y_true = outGT.data.cpu().numpy().ravel()  
            y_pred_prob = outPRED_sigmoid.data.cpu().numpy().ravel()  

            
            y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

           
            f1 = f1_score(y_true, y_pred)
            self.epochs_stats['f1 val'].append(f1)
            print("F1 Score:", f1)

            self.epochs_stats['loss val'].append(each_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            self.epochs_stats['macro_auroc val'].append(ret)
            self.epochs_stats['weighted_auroc val'].append(weighted_auc)

       
            self.epochs_stats['micro_auroc val'].append(micro_auc)

           
            self.epochs_stats['auprc val'].append(pr_auc)
            for i in range(outGT.shape[1]):  
                class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
                # print(f'train AUC for class {i + 1}: {class_auroc}')
                
               
                self.epochs_stats[f'auroc {i + 1} val'].append(class_auroc)

            return ret

    def save_epochs_stats(self, file_path='result.txt'):
   
        # print(f'-------------------{self.epochs_stats}----------')
        epochs_stats=self.epochs_stats
        for key, value in epochs_stats.items():
            if isinstance(value, list):
                epochs_stats[key] = [v.item() if isinstance(v, torch.Tensor) else v for v in value]
            elif isinstance(value, torch.Tensor):
                epochs_stats[key] = value.item()

        with open(file_path, 'w') as f:
            json.dump(self.epochs_stats, f, indent=4)


    def test(self):
        print(f'-----------------starting test--------------------')
        test_loss = 0
        self.load_best_checkpoint()
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        each_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (ecg_data, cxr_data, cli,target,_,_) in enumerate(self.val_dl):
               
                ecg_data_np = np.array(ecg_data)


                ecg_data = torch.from_numpy(ecg_data_np).float()
                ecg_data = ecg_data.float()
                # cxr_data = cxr_data.to(self.device)
                target = torch.from_numpy(target).float()

                ecg_data= Variable(ecg_data.to(self.device),requires_grad=False)
                cxr_data= Variable(cxr_data.to(self.device),requires_grad=False)
               
                target= Variable(target.to(self.device),requires_grad=False)


                text_outputs, image_outputs, output, total_loss = self.model(ecg_data, cxr_data,cli)



                loss = 0.8*self.loss(output,target)
            # print
                total_loss=total_loss.mean()
            # print(f'loss {loss}')
            # print(f'total loss {total_loss}')
                loss+=0.2*total_loss.item()
                each_loss+=loss.item()

                # loss=self.loss(output,target)
                # # print(f'each val loss{loss}')
                # epoch_loss+=loss.item()
                # total_loss=total_loss.mean()
                # epoch_loss+=total_loss.item()
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

            
            print(f"lr {self.args.lr} val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{each_loss/batch_idx:0.5f} ")
            # ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            outPRED_sigmoid = torch.sigmoid(outPRED)
            # outPRED_sigmoid=outPRED
            ret=roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='macro')
            print(f'val AUC {ret}')
            self.scheduler.step(ret)
            # precision, recall, _ = precision_recall_curve(outGT.data.cpu().numpy().ravel(), outPRED_sigmoid.data.cpu().numpy().ravel())
            # pr_auc = auc(recall, precision)
            micro_auc = roc_auc_score(outGT.data.cpu().numpy(), outPRED_sigmoid.data.cpu().numpy(), average='micro')


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
            y_true = outGT.data.cpu().numpy().ravel() 
            y_pred_prob = outPRED_sigmoid.data.cpu().numpy().ravel() 

           
            y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

           
            f1 = f1_score(y_true, y_pred)
            self.epochs_stats['f1 val'].append(f1)
            print("F1 Score:", f1)

            self.epochs_stats['loss val'].append(each_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            self.epochs_stats['macro_auroc val'].append(ret)
            self.epochs_stats['weighted_auroc val'].append(weighted_auc)

        # 'micro_auroc train': [], 'weighted_auroc train': []
            self.epochs_stats['micro_auroc val'].append(micro_auc)

            # self.epochs_stats['loss val'].append(epoch_loss/batch_idx)
            # self.epochs_stats['auroc val'].append(ret)
            self.epochs_stats['auprc val'].append(pr_auc)
            for i in range(outGT.shape[1]): 
                class_auroc = roc_auc_score(outGT[:, i].data.cpu().numpy(), outPRED_sigmoid[:, i].data.cpu().numpy())
                # print(f'train AUC for class {i + 1}: {class_auroc}')
                
                
                self.epochs_stats[f'auroc {i + 1} val'].append(class_auroc)


            file_path = f'result_record/G9_test_{args.fusion_type}_{args.fusion_model}_{args.name}_result.txt'
            self.save_epochs_stats(file_path)

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

            file_path = f'result_record/G9_{args.name}_{args.fusion_type}_{args.fusion_model}_result.txt'
            self.save_epochs_stats(file_path)



            
    



            
            
