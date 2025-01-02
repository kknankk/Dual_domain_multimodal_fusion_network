import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class LSTM(nn.Module):

    def __init__(self, input_dim=12, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()
        self.activation = torch.sigmoid
    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x=x.permute(0,2,1)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        out = self.dense_layer(feats)
        scores = torch.sigmoid(out)
        return scores, feats






import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax

import torch.nn.functional as F

class CXRModels(nn.Module):

    def __init__(self, args, device='cpu'):
	
        super(CXRModels, self).__init__()
        self.args = args
        self.device = device
        self.vision_backbone = getattr(torchvision.models, 'resnet34')(pretrained=True)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, 4))
        self.feats_dim = d_visual
       

    def forward(self, x, labels=None, n_crops=0, bs=16):
        lossvalue_bce = torch.zeros(1).to(self.device)
        # print(f'cxr input{x.shape}')
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        preds = torch.sigmoid(preds)

        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)

        return preds, lossvalue_bce, visual_feats
    

  
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F

class medfuse(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(medfuse, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        target_classes = 4
        # lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, 4),
            nn.Sigmoid()
        )

        # self.align_loss = CosineLoss()
        # self.kl_loss = KLDivLoss()



        

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)


        # if self.args.labels_set == 'radiology':
        #     target_classes = self.args.vision_num_classes
        #     lstm_in = self.cxr_model.feats_dim
        #     projection_in = self.ehr_model.feats_dim

        # import pdb; pdb.set_trace()
        # self.projection = nn.Linear(projection_in, 256)
        # # feats_dim = 2 * self.ehr_model.feats_dim
        # # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        # self.fused_cls = nn.Sequential(
        #     nn.Linear(512, 4),
        #     nn.Sigmoid()
        # )

        # # self.align_loss = CosineLoss()
        # # self.kl_loss = KLDivLoss()



        

        # self.lstm_fused_cls =  nn.Sequential(
        #     nn.Linear(lstm_out, target_classes),
        #     # nn.Sigmoid()
        # ) 

        # self.lstm_fusion_layer = nn.LSTM(
        #     256, lstm_out,
        #     batch_first=True,
        #     dropout = 0.0)
        # self.lin1 = nn.Linear(512, 256)

            
    # def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
    #     cxr_preds, _ , feats = self.cxr_model(img)
    #     return {
    #         'uni_cxr': cxr_preds,
    #         'cxr_feats': feats
    #         }
    # 
    # def forward(self, x, seq_lengths=None, img=None, pairs=None ):
    #     if self.args.fusion_type == 'uni_cxr':
    #         return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
    #     elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified']:
    #         return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
    #     elif self.args.fusion_type == 'uni_ehr':
    #         return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
    #     elif self.args.fusion_type == 'lstm':
    #         return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

    #     elif self.args.fusion_type == 'uni_ehr_lstm':
    #         return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

    # def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
    #     ehr_preds , feats = self.ehr_model(x, seq_lengths)
    #     return {
    #         'uni_ehr': ehr_preds,
    #         'ehr_feats': feats
    #         }

    # def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):

    #     ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
    #     cxr_preds, _ , cxr_feats = self.cxr_model(img)
    #     projected = self.projection(cxr_feats)

    #     # loss = self.align_loss(projected, ehr_feats)

    #     feats = torch.cat([ehr_feats, projected], dim=1)
    #     fused_preds = self.fused_cls(feats)

    #     # late_avg = (cxr_preds + ehr_preds)/2
    #     return {
    #         'early': fused_preds, 
    #         'joint': fused_preds, 
    #         # 'late_avg': late_avg,
    #         # 'align_loss': loss,
    #         'ehr_feats': ehr_feats,
    #         'cxr_feats': projected,
    #         'unified': fused_preds
    #         }


#==========将forward_lstm_fused改为forward=========
    def initial_forward(self, x, seq_lengths=4096, img=None, pairs=None ):
        # if self.args.labels_set == 'radiology':
        #     _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
        #     _, _ , cxr_feats = self.cxr_model(img)

        #     feats = cxr_feats[:,None,:]

        #     ehr_feats = self.projection(ehr_feats)

        #     ehr_feats[list(~np.array(pairs))] = 0
        #     feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        # else:
        # print(f'fusion input ecg {x.shape}')#[batch,12,4096]
        # x=x.permute(0,2,1)
        # print(f'fusion input ecg {x.shape}')
        _,ehr_feats = self.ehr_model(x,seq_lengths)
        # if 
        # print(f'initial ehr_feats {ehr_feats.shape}')#[bs,256]
        
        _, _ , cxr_feats = self.cxr_model(img)
        cxr_feats = self.projection(cxr_feats)
        # ehr_feats=self.lin1(ehr_feats)
        # print(f'ecg after lin {ehr_feats.shape}')

        # false_count = pairs.count(False)
        false_count = torch.sum(pairs == False).item()  
        if false_count > 0:
            print(f"========================There are {false_count} False values in pairs.================================")
        # print(f'ehr {ehr_feats.shape}')
        # print(f'cxr_feats {cxr_feats.shape}')
        # cxr_feats[list(~np.array(pairs))] = 0
        # print(f'medfuse len(ehr_feats.shape) {ehr_feats.shape}')
        if len(ehr_feats.shape) == 1:
            # print(ehr_feats.shape, cxr_feats.shape)
            # import pdb; pdb.set_trace()
            feats = ehr_feats[None,None,:]
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        else:
            feats = ehr_feats[:,None,:]
            # print(f'select ecg {feats.shape}')
            # cxr_feats1=cxr_feats[:,None,:]
            # print(f'cxr_feats1 {cxr_feats.shape}')
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            # print(f' concat {feats.shape}')
        # seq_lengths = np.array([1] * len(seq_lengths))
        # seq_lengths[pairs] = 2
        
        # feats = torch.nn.utils.rnn.pack_padded_sequence(feats, 4096, batch_first=True, enforce_sorted=False)
        # print(f'after cat feats{feats.shape}')
        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)
        # print(f'output in model {fused_preds.shape}')
        if fused_preds.dim() == 1:
            fused_preds = fused_preds.unsqueeze(0)  
        # return {
        #     'lstm': fused_preds,
        #     'ehr_feats': ehr_feats,
        #     'cxr_feats': cxr_feats,
        # }
        return fused_preds

    def forward(self, x, seq_lengths=4096, img=None, pairs=None ):

        x=x.permute(0,2,1)
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        # if 

        _, _ , cxr_feats = self.cxr_model(img)
        cxr_feats = self.projection(cxr_feats)

        # cxr_feats[list(~np.array(pairs))] = 0
        if len(ehr_feats.shape) == 1:
        # print(ehr_feats.shape, cxr_feats.shape)
        # import pdb; pdb.set_trace()
            feats = ehr_feats[None,None,:]
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        else:
            feats = ehr_feats[:,None,:]
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        # seq_lengths = np.array([1] * len(seq_lengths))
        # seq_lengths[pairs] = 2

        # feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()

        fused_preds = self.lstm_fused_cls(out)

        return fused_preds

    # def forward_lstm_ehr(self, x, seq_lengths=4096, img=None, pairs=None ):
    #     print(f'fusion model input ecg {x.shape}')
    #     _ , ehr_feats = self.ehr_model(x, seq_lengths)
    #     feats = ehr_feats[:,None,:]
        
        
    #     # seq_lengths = np.array([1] * len(seq_lengths))
        
    #     # feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

    #     x, (ht, _) = self.lstm_fusion_layer(feats)

    #     out = ht.squeeze()
        
    #     fused_preds = self.lstm_fused_cls(out)

    #     return {
    #         'uni_ehr_lstm': fused_preds,
    #     }
