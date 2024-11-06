from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score
# torch.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=np.inf)
def evaluate_new(df):
    auroc = roc_auc_score(df['y_truth'], df['y_pred'])
    auprc = average_precision_score(df['y_truth'], df['y_pred'])
    return auprc, auroc



def bootstraping_eval(df, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    auroc_list = []
    auprc_list = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
    return auprc_list, auroc_list

def computing_confidence_intervals(list_,true_value):
    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    # print(f"CI 95% {round(true_value, 3)} ( {round(lower, 3)} , {round(upper, 3)} )")
    return (upper,lower)


def get_model_performance(df):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=1000)
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc)



class Trainer():
    def __init__(self, args):
        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0

    def train(self):
        pass

    def train_epoch(self):
        pass

    def validate(self):
        pass

    def plot_array(self, array, disc='loss'):
        plt.plot(array)
        plt.ylabel(disc)
        plt.savefig(f'{disc}.pdf')
        plt.close()

    def get_gt(self, y):
        return torch.from_numpy(y).float()

    def computeAUROC(self, y_true, predictions, verbose=1):
        y_true = np.array(y_true)
        predictions = np.array(predictions)
        print(f'prediction in computeauroc {predictions}')
        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")

        auprc = metrics.average_precision_score(y_true, predictions, average=None)
        
        
        auc_scores = []
        auprc_scores = []
        ci_auroc = []
        ci_auprc = []
        if len(y_true.shape) == 1:
            y_true = y_true[:, None]
            predictions = predictions[:, None]
        for i in range(y_true.shape[1]):
            df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': predictions[:, i]})
            (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc) = get_model_performance(df)
            auc_scores.append(test_auroc)
            auprc_scores.append(test_auprc)
            ci_auroc.append((lower_auroc, upper_auroc))
            ci_auprc.append((lower_auprc, upper_auprc))
        
        auc_scores = np.array(auc_scores)
        auprc_scores = np.array(auprc_scores)
       
        return { "auc_scores": auc_scores,
            
            "auroc_mean": np.mean(auc_scores),
            "auprc_mean": np.mean(auprc_scores),
            "auprc_scores": auprc_scores, 
            'ci_auroc': ci_auroc,
            'ci_auprc': ci_auprc,
            }

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{self.args.fusion_type}/{self.args.domain}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'best_auroc': self.best_auroc, 
            'optimizer' : self.optimizer.state_dict(),
            'epochs_stats': self.epochs_stats
            }, path)
        print(f"saving {prefix} checkpoint at epoch {self.epoch}")

    # def print_and_write(self,ret,prefix='val',isbest=False,filename='results.txt'):

#============dr_fuse loss===========================
import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_auroc

import lightning.pytorch as pl
def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

def _masked_abs_cos_sim(self, x, y, mask):
    return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

def _masked_cos_sim(self, x, y, mask):
    return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

def _masked_mse(self, x, y, mask):
    return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
    ehr_mask = torch.ones_like(pairs)
    loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                            model_output['feat_cxr_distinct'], pairs)
    loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                            model_output['feat_ehr_distinct'], ehr_mask)

    jsd = self.jsd(model_output['feat_ehr_shared'].sigmoid(),
                model_output['feat_cxr_shared'].sigmoid(), pairs)

    loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                            self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                            self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
    if log:
        self.log_dict({
            f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
            f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
            f'disentangle_{mode}/shared_jsd': jsd.detach(),
            'step': float(self.current_epoch)
        }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

    return loss_disentanglement

def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
    ehr_mask = torch.ones_like(model_output['pred_final'][:, 0])
    loss_pred_final = self._compute_masked_pred_loss(model_output['pred_final'], y_gt, ehr_mask)
    loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
    loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
    loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)

    if log:
        self.log_dict({
            f'{mode}_loss/pred_final': loss_pred_final.detach(),
            f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
            f'{mode}_loss/pred_ehr': loss_pred_ehr.detach(),
            f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
            'step': float(self.current_epoch)
        }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

    return loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared

def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
    prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
    loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared = prediction_losses

    loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                    self.hparams.lambda_pred_ehr * loss_pred_ehr +
                    self.hparams.lambda_pred_cxr * loss_pred_cxr)

    loss_prediction = loss_pred_final + loss_prediction

    loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

    loss_total = loss_prediction + loss_disentanglement
    epoch_log = {}

    # aux loss for attention ranking
    raw_pred_loss_ehr = F.binary_cross_entropy(model_output['pred_ehr'].data, y_gt, reduction='none')
    raw_pred_loss_cxr = F.binary_cross_entropy(model_output['pred_cxr'].data, y_gt, reduction='none')
    raw_pred_loss_shared = F.binary_cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')

    pairs = pairs.unsqueeze(1)
    attn_weights = model_output['attn_weights']
    attn_ehr, attn_shared, attn_cxr = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2]

    cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
    loss_attn1 = pairs * F.margin_ranking_loss(attn_cxr, attn_ehr, cxr_overweights_ehr, reduction='none')
    loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1>0].numel())

    shared_overweights_ehr = 2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
    loss_attn2 = pairs * F.margin_ranking_loss(attn_shared, attn_ehr, shared_overweights_ehr, reduction='none')
    loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2>0].numel())

    shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
    loss_attn3 = pairs * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
    loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3>0].numel())

    loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3) / 3

    loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking
    epoch_log[f'{mode}_loss/attn_aux'] = loss_attn_ranking.detach()

    if log:
        epoch_log.update({
            f'{mode}_loss/total': loss_total.detach(),
            f'{mode}_loss/prediction': loss_prediction.detach(),
            'step': float(self.current_epoch)
        })
        self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

    return loss_total
