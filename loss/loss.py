import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.functional import cosine_similarity


class BatchBasedClassificationLoss(nn.Module):
    def __int__(self):
        super().__init__()

    def cos_sim(self, a, b):
        return cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)

    def forward(self, predicted_features, tar_features):
        prediction = 100 * predicted_features @ tar_features.T
        # prediction = self.cos_sim(predicted_features, tar_features)
        labels = torch.arange(0, predicted_features.size(0)).long()

        return F.cross_entropy(prediction, labels)


class MarginKLLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.alpha = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.beta = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, a, b):
        y = torch.ones_like(a)
        margin_loss = self.alpha * F.margin_ranking_loss(a, b, y, margin=self.margin)

        kl_loss = self.beta * (F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b, dim=-1), reduction='batchmean') +
                               F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a, dim=-1), reduction='batchmean'))

        total_loss = margin_loss + kl_loss * 0
        return total_loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.4):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.alpha = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, a, b):
        # compute pairwise distances between all samples in batch
        positive_a = a
        idx = torch.randperm(a.size(0))
        negative_a = a[idx]
        anchor_b = b

        positive_b = b
        idx = torch.randperm(a.size(0))
        negative_b = b[idx]
        anchor_a = a

        # compute triplet loss
        loss_trip_a = self.triplet_loss(anchor_a, positive_b, negative_b)
        loss_trip_b = self.triplet_loss(anchor_b, positive_a, negative_a)

        return self.alpha * (loss_trip_a + loss_trip_b)


class MyLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MyLoss, self).__init__()
        self.margin = margin
        self.cos_loss = nn.CosineEmbeddingLoss(margin=self.margin)
        self.margin_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, a, b):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(a, b)

        # 计算余弦相似度的损失
        cos_loss = self.cos_loss(a, b, torch.ones_like(cos_sim))

        # 计算距离损失
        dist = torch.norm(a - b, p=2, dim=1)  # 计算欧几里得距离
        margin_target = torch.ones_like(dist)  # 目标为1
        margin_loss = self.margin_loss(dist, dist, margin_target)

        print(cos_loss)
        print(margin_loss)

        # 组合损失
        loss_total = cos_loss + margin_loss

        return loss_total


if __name__ == '__main__':
    loss = TripletLoss()
    a, b = torch.randn([32, 640]), torch.randn([32, 640])
    c = loss(a, b)
    print(c)

