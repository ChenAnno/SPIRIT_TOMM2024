import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import math
import numpy as np
from aoa_pytorch import AoA


class SelfAttentionNew(nn.Module):
    def __init__(self, dim=640, heads=8, dim_head=80):
        super(SelfAttentionNew, self).__init__()
        self.aoa = AoA(dim=dim, heads=heads, dim_head=dim_head)

    def forward(self, feature):
        """
        AoA self-attention:An Improved Attention for Visual Question Answering
        :param feature:[b, 640]
        :return:[b, 640]
        """
        feature_ex = torch.unsqueeze(feature, dim=1)  # [b, 1, 640]
        feature_aoa = self.aoa(feature_ex) + feature_ex  # [b, 1, 640]
        feature_aoa = torch.squeeze(feature_aoa, dim=1)  # [b, 640]
        return feature_aoa


class GuidedAttentionNew(nn.Module):
    def __init__(self, dim=640, heads=8, dim_head=80):
        super(GuidedAttentionNew, self).__init__()
        self.aoa = AoA(dim=dim, heads=heads, dim_head=dim_head)

    def forward(self, feature, context):
        """
        AoA self-attention:An Improved Attention for Visual Question Answering
        :param feature:[b, 1280]
        :param context:[b, 1280]
        :return:[b, 1280]
        """
        feature_ex = torch.unsqueeze(feature, dim=1)  # [b, 1, 1280]
        context = torch.unsqueeze(context, dim=1)  # [b, 1, 1280]
        feature_aoa = self.aoa(feature_ex, context=context) + feature_ex  # [b, 1, 1280]
        feature_aoa = torch.squeeze(feature_aoa, dim=1)  # [b, 1280]
        return feature_aoa


class CombinerAoA(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, feature_dim=1280):
        super(CombinerAoA, self).__init__()

        self.image_mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim * 4),  # 这是一个超参数可以调
                                       nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(feature_dim * 4, 1), nn.Softmax(dim=-1))
        self.text_mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim * 4),  # 这是一个超参数可以调
                                      nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(feature_dim * 4, 1), nn.Softmax(dim=-1))
        self.fusion_concat = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),  # 这是一个超参数可以调
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1))

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: scaled logits
        """
        b = image_features.shape[0]
        image_attention = self.image_mlp(image_features).reshape(b, -1).expand_as(image_features) * image_features
        text_attention = self.text_mlp(text_features).reshape(b, -1).expand_as(text_features) * text_features
        concat_feature = torch.cat((image_attention, text_attention), dim=-1)  # [b, 2048]
        fusion_score = self.fusion_concat(concat_feature)  # [b, 2]
        image_fusion = fusion_score[:, 0].reshape(b, -1).expand_as(image_attention) * image_attention
        text_fusion = fusion_score[:, 1].reshape(b, -1).expand_as(text_attention) * text_attention
        total_fusion = image_fusion + text_fusion + image_attention + text_attention
        total_fusion = F.normalize(total_fusion, dim=-1)  # TODO 注意这里没有用 LayerNorm
        return total_fusion


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 640)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 640)
    """

    def __init__(self, sim_dim=640):
        super(AttentionFiltration, self).__init__()
        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)
        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MHAtt(nn.Module):
    def __init__(self, hidden_size=1280, multi_head=16, hidden_size_head=80, dropout=0.1):
        """
        :param hidden_size: [b, feature size]
        :param multi_head:
        :param hidden_size_head: 这两个相乘应该等于hidden_size
        :param dropout:
        """
        super(MHAtt, self).__init__()
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.multi_head = multi_head
        self.hidden_size_head = hidden_size_head
        self.hidden_size = hidden_size

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        attn = self.att(v, k, q, mask)
        attn = attn.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )
        attn = self.linear_merge(attn)
        return attn

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FFN(nn.Module):
    def __init__(self, hidden_size=1280, ff_size=1280 * 2, dropout=0.1):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, hidden_size=1280, ff_size=1280 * 2, multi_head=16, hidden_size_head=80, dropout=0.1):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size, multi_head, hidden_size_head, dropout)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class SGA(nn.Module):
    def __init__(self, hidden_size=1280, ff_size=1280 * 2, multi_head=16, hidden_size_head=80, dropout=0.1):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size, multi_head, hidden_size_head, dropout)
        self.mhatt2 = MHAtt(hidden_size, multi_head, hidden_size_head, dropout)
        self.ffn = FFN(hidden_size, ff_size)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class Fusion_Attention(nn.Module):
    def __init__(self, hidden_size=1280, ff_size=1280 * 2, multi_head=16, hidden_size_head=80, dropout=0.1):
        super(Fusion_Attention, self).__init__()
        # local trans
        self.l2l_SA = SA(hidden_size, ff_size, multi_head, hidden_size_head, dropout)
        # global trans
        self.g2g_SA = SA(hidden_size, ff_size, multi_head, hidden_size_head, dropout)
        # local correction
        self.g2l_SGA = SGA(hidden_size, ff_size, multi_head, hidden_size_head, dropout)
        # global supplement
        self.l2g_SGA = SGA(hidden_size, ff_size, multi_head, hidden_size_head, dropout)
        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),  # 这里的dropout不同
            nn.Linear(hidden_size, 2),
            nn.Softmax()
        )

    def forward(self, local_feature, global_feature):
        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)
        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)
        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)
        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)
        global_feature = torch.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0], -1).expand_as(global_feature)
        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0], -1).expand_as(global_feature)

        visual_feature = weight_global * global_feature + weight_local * local_feature
        return visual_feature


if __name__ == '__main__':
    # a = torch.randn([2, 20, 1024])
    # b = torch.randn([2, 1024])
    # local_net = LocalNet()
    # c = local_net(a)
    # print(c.shape)

    # f = torch.randn([2, 640])
    # aoa = SelfAttentionNew()
    # out = aoa(f)
    # print(out.shape)

    image_f, text_f = torch.randn([2, 1280]), torch.randn([2, 1280])
    aoa = GuidedAttentionNew()
    out = aoa(image_f, text_f)
    print(out.shape)

