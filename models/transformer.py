from collections import OrderedDict
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from clip.model import QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiFrameIntegrationTransformer(nn.Module):
    def __init__(self, T=13, embed_dim=640, layers=1, fp16=True):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 80
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])
        self.apply(self._init_weights)

        self.fp16 = fp16

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            ori_x = x
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.resblocks(x)
            x = x.permute(1, 0, 2)
            x = x.type(ori_x.dtype) + ori_x
            if self.fp16:
                return x.mean(dim=1, keepdim=False).float()
            return x.mean(dim=1, keepdim=False)


if __name__ == "__main__":
    model = MultiFrameIntegrationTransformer(13, 640)
    features = torch.randn([2, 13, 640])
    out = model(features)

    print(out.shape)

