""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
    # def info_nce_loss(self, q, q_new):

    #     labels = torch.cat([torch.arange(49) for i in range(1)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(q.get_device())

    #     features_q = F.normalize(q, dim=1)
    #     features_q_new = F.normalize(q_new, dim=1)

    #     similarity_matrix = torch.matmul(features_q, features_q_new.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(q.get_device())
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.get_device())

    #     logits = logits / 0.1
    #     return logits, labels

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x0, q, q_new = x
        logprobs = F.log_softmax(x0, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        loss_cluster = nll_loss.mean()*0
        labels = torch.cat([torch.arange(49) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(q[0].get_device())
        labels = labels.repeat(q[0].shape[0], 1, 1)
        bs = x0.size()[0]
        num_q = q[0].size()[-2]
        for i in range(len(q)):
            # loss = self.info_nce_loss(q[i], q_new[i])
            scores = torch.matmul(q[i], q_new[i].transpose(-1, -2)) / torch.sqrt(torch.tensor(q_new[i].shape[-1], dtype=torch.float32))
            loss_cluster = loss_cluster + F.cross_entropy(scores.reshape(-1,49),labels.reshape(-1,49))/bs
            # loss_cluster = loss_cluster - F.cosine_similarity(q[i], q_new[i]).abs().mean()      
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss 
        return loss.mean()+ loss_cluster /len(q)

    
    
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
