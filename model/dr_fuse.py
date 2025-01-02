import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

# from .ehr_transformer import EHRTransformer
import torch
from torch import nn

#adaptive from 'https://github.com/dorothy-yao/drfuse'
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


class EHRTransformer(nn.Module):
    #=============change max_len从350改为4-96
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=4097):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        # attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
        #                          float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
        #                         for len_ in seq_lengths])
        # print(f'ecg transform inout ecg {x.shape}')#[16,12,4096]
        x=x.permute(0,2,1)
        # print(f'after permute ecg {x.shape}')

        x = self.emb(x) # [16,4096,12]
        x = self.pos_encoder(x)

        return rep_shared, rep_distinct, pred_distinct



class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head,
                 cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
      
        self.ehr_model = EHRTransformer(input_size=12, num_classes=num_classes,
                                        d_model=hidden_size, n_head=ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                        n_layers_distinct=ehr_n_layers,
                                        dropout=ehr_dropout)

        resnet = resnet50()
        self.cxr_model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        resnet = resnet50()
        self.cxr_model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_shared.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        resnet = resnet50()
        self.cxr_model_spec = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_spec.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.cxr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, img, seq_lengths, pairs):

        # x=x.permute(0,2,1)
        # print(f'after permute ecg {x.shape}')
        
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model_feat(img)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)

        # get shared feature
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)
        # pairs = torch.FloatTensor(pairs)
        pairs = pairs.unsqueeze(1)
        # print(f' after unqueeze pairs  {pairs}')
        h1 = feat_ehr_shared
        h2 = feat_cxr_shared
        term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        # print(f'pairs {pairs.shape}')
        # print(f'feat_avg_shared {feat_avg_shared.shape}')
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Disease-wise Attention
        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1-pairs) * q[:, :-1].mean(dim=1)

        # compute attention weighting
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        # print(f'pairs in drfuse model {pairs}')
        # if pairs.squeeze()==0:
        if (pairs.squeeze()).any()==0:
            print(f'pairs.squeeze()==0')
        attn_mask[pairs.squeeze()==0, :, -1] = 0
        
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)

        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()
        # pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)

        outputs = {
            'feat_ehr_shared': feat_ehr_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
        }

        return outputs

