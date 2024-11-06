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
        # b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = x.view(x.size(0), 8, 1, 64)  # b, head=8, c=1, (h w)=64
        k = x.view(x.size(0), 8, 1, 64)  # b, head=8, c=1, (h w)=64
        v = x.view(x.size(0), 8, 1, 64) 
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
        # ori = spa
        # fre = self.fre(fre)
        # spa = self.spa(spa)
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        return F.relu(self.layers(inputs) + inputs)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act,norm, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResBlock(n_feat) for _ in range(n_resblocks)]

        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)



class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.act = None
        self.norm = None

        self.act = torch.nn.PReLU()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
        

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
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        out = self.dense_layer(feats)
        scores = torch.sigmoid(out)
        return scores, feats



# # 使用输入形状 [16, 4096, 12]
# input_tensor = torch.randn(16, 4096, 12)
# # seq_lengths = torch.randint(1, 4096, (16,))  # 随机生成序列长度
# output, features = LSTM(seq_lengths=4096)


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

        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        preds = torch.sigmoid(preds)

        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)

        return preds, lossvalue_bce, visual_feats
    

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, _, _ = x.size()
        
        # 计算 Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (x.size(-1) ** 0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        # 计算加权的值
        attention_output = torch.matmul(attention_weights, v)

        return attention_output

  
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F

class mod_medfuse(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(mod_medfuse, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model


        target_classes = 4
        # lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        

        # if self.args.labels_set == 'radiology':
        #     target_classes = self.args.vision_num_classes
        #     lstm_in = self.cxr_model.feats_dim
        #     projection_in = self.ehr_model.feats_dim

        # import pdb; pdb.set_trace()
        self.projection = nn.Linear(projection_in, 256)
        # feats_dim = 2 * self.ehr_model.feats_dim
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        self.self_attention = SelfAttention(in_channels=256)
        self.self_attention1 = SelfAttention(in_channels=256)
        self.self_attention2 = SelfAttention(in_channels=256)  # 第二层自注意力
        self.fc1 = nn.Linear(768, 256)  # 输入维度为 512（模态 A 和 B 的拼接）
        self.fc2 = nn.Linear(256, 4)  # 4 分类输出
        self.dropout = nn.Dropout(0.3)  # Dropout 层，防止过拟合

        # self.fc = nn.Linear(256, 4)
        # self.align_loss = CosineLoss()
        # self.kl_loss = KLDivLoss()



        

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            # nn.Sigmoid()
        ) 

        self.lstm_fusion_layer = nn.LSTM(
            12, lstm_out,
            batch_first=True,
            dropout = 0.0)

        self.lin1 = nn.Linear(512, 256)

        self.conv_fuse=FuseBlock7(64)
        self.up3_mo = nn.Sequential(ResidualGroup(
            64, 3, 4, act=None, n_resblocks=2, norm=None))

        # define tail module
        modules_tail = [
            ConvBNReLU2D(64, out_channels=3, kernel_size=3, padding=1,
                         act=None)]

        self.tail = nn.Sequential(*modules_tail)


            
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
    def forward(self, x, seq_lengths=4096, img=None, pairs=None ):
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
        # print(f'input mod_medfuse ecg {x.shape}')
        x=x.transpose(2,1)
        ehr_feats,_   = self.ehr_model(x)
        # if 
        # print(f'resnet1d ecg {ehr_feats.shape}')#[bs.512]

        ehr_feats=self.lin1(ehr_feats)

        
        _, _ , cxr_feats = self.cxr_model(img)
        # print(f'cxr1 {cxr_feats.shape}')
        cxr_feats = self.projection(cxr_feats)
        # print(f'cxr {cxr_feats.shape}')

        # false_count = pairs.count(False)
        false_count = torch.sum(pairs == False).item()  
        if false_count > 0:
            print(f"========================There are {false_count} False values in pairs.================================")
        # print(f'ehr {ehr_feats.shape}') #[bs,1,256]
        # print(f'cxr_feats {cxr_feats.shape}')#[bs,256]
        # cxr_feats[list(~np.array(pairs))] = 0
        # print(f'medfuse len(ehr_feats.shape) {ehr_feats.shape}')
        if len(ehr_feats.shape) == 1:
            # print(ehr_feats.shape, cxr_feats.shape)
            # import pdb; pdb.set_trace()
            feats1 = ehr_feats[None,None,:]
            feats = torch.cat([feats1, cxr_feats[:,None,:]], dim=1)
        else:
            feats1 = ehr_feats[:,None,:]
            # cxr_feats=cxr_feats[:,None,:]
            # print(f'select ecg {feats.shape}')#[bs,256]
            feats = torch.cat([feats1, cxr_feats[:,None,:]], dim=1)#[bs,2,256]
            
        # print(f'final feats {feats.shape}')
        attention_output1 = self.self_attention1(feats)  # 形状为 [batch_size, 2, 256]
        attention_output2 = self.self_attention2(attention_output1)  # 第二层自注意力

        # 取最后一个时间步的输出
        attention_output = attention_output2[:, -1, :]  # 形状为 [batch_size, 256]

        # 特征融合
        # 拼接模态 A 和 B 的输出
        # print(f'cxr_feats {cxr_feats.shape}')
        # print(f'ecg_feats {feats1.shape}')
        combined_features = torch.cat((cxr_feats[:, :], feats1[:, 0, :], attention_output), dim=1)  # 形状为 [batch_size, 512]

        # 分类输出
        x = self.fc1(combined_features)  # 形状为 [batch_size, 256]
        x = nn.functional.relu(x)  # 激活函数
        x = self.dropout(x)  # Dropout
        output = self.fc2(x)  # 形状为 [batch_size, 4]
        return output


        # seq_lengths = np.array([1] * len(seq_lengths))
        # seq_lengths[pairs] = 2
        
        # feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)
        # print(f'ecg feats {feats.shape}')#[bs,1,256]
        # x, (ht, _) = self.lstm_fusion_layer(feats)
        # print(f'cxr feats {cxr_feats.shape}')
        # feats=feats.transpose(2,0)
        # cxr_feats=cxr_feats.transpose(2,0)
        # print(f'cxr feats {cxr_feats.shape}')
        # print(f'ecg feats {feats.shape}')
        # up3_fuse = self.conv_fuse(feats, cxr_feats)
        # up3_mo = self.up3_mo(up3_fuse)

        # up3_mo = up3_mo + x
        # up3_fuse_mo = self.conv_fuse(up3_fre_mo, up3_mo)

        # res = self.tail(up3_fuse_mo)


        # out = ht.squeeze()
        
        # fused_preds = self.lstm_fused_cls(out)
        # print(f'output in model {fused_preds.shape}')
        # if fused_preds.dim() == 1:
        #     fused_preds = fused_preds.unsqueeze(0)  # 转换为 [1, n] 的形状
        # # return {
        # #     'lstm': fused_preds,
        # #     'ehr_feats': ehr_feats,
        # #     'cxr_feats': cxr_feats,
        # # }
        # return fused_preds

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