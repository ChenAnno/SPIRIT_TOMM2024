import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.clip_model import ImageCLIP, TextCLIP
from models.attention import SelfAttentionNew, AttentionFiltration
from models.transformer import MultiFrameIntegrationTransformer
from loss.loss import TripletLoss


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(
            hidden_dim, clip_feature_dim * 2
        )  # [batch_size, 1280]

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: scaled logits
        """

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * image_features
        )

        predicted_features = F.normalize(output, dim=-1)
        return predicted_features


class Combiner2(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner2, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)

        self.image_mlp = nn.Sequential(
            nn.Linear(clip_feature_dim * 2, clip_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(clip_feature_dim * 2, 1),
            nn.Softmax(dim=-1),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(clip_feature_dim * 2, clip_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(clip_feature_dim * 2, 1),
            nn.Softmax(dim=-1),
        )

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim * 2)  # [batch_size, 1280]
        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout11 = nn.Dropout(0.5)
        self.dropout22 = nn.Dropout(0.5)
        self.combiner_layer2 = nn.Linear(projection_dim, hidden_dim)
        self.output_layer2 = nn.Linear(
            hidden_dim, clip_feature_dim * 2
        )  # [batch_size, 1280]
        self.dropout33 = nn.Dropout(0.5)
        self.dynamic_scalar2 = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: scaled logits
        """

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))  # [b, 640*4]
        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)  # [b, 640*8]
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))  # [b, 640*8]
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        text_sa = self.dropout11(F.relu(text_features * self.text_mlp(text_features)))  # [b, 640*2]
        image_sa = self.dropout11(F.relu(image_features * self.text_mlp(image_features)))
        raw_combined_features2 = torch.cat((text_sa, image_sa), -1)  # [b, 640*4]
        combined_feature2 = self.dropout33(F.relu(self.combiner_layer2(raw_combined_features2)))  # [b, 640*8]
        dynamic_scalar2 = self.dynamic_scalar2(raw_combined_features2)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * image_features
            + self.output_layer2(combined_feature2)
            + dynamic_scalar2 * text_sa
            + (1 - dynamic_scalar2) * image_sa
        )
        predicted_features = F.normalize(output, dim=-1)
        return predicted_features


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 13, 640)
          - raw_global: raw image by averaging regions, shape: (batch_size, 640)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 640).
    """

    def __init__(self, embed_dim=640, dropout_rate=0.5, num_region=13):
        super(VisualSA, self).__init__()
        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

        self.self_attention = SelfAttentionNew()

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def l2norm(self, X, dim=-1, eps=1e-8):
        """L2-normalize columns of X"""
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    def forward(self, local_feature):
        raw_global = torch.mean(local_feature, 1)
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local_feature)
        g_emb = self.embedding_global(raw_global)
        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)
        # compute final image, shape: (batch_size, 640)
        new_global = (weights.unsqueeze(2) * local_feature).sum(dim=1)

        new_global = self.self_attention(new_global)
        return self.l2norm(new_global, dim=-1)


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 640)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 640)
    """

    def __init__(self, sim_dim=640):
        super(GraphReasoning, self).__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VisualGR(nn.Module):
    def __init__(self, embed_dim=640, graph_num=4):
        super(VisualGR, self).__init__()
        self.graph = nn.ModuleList(
            [GraphReasoning(sim_dim=embed_dim) for _ in range(graph_num)]
        )
        self.attention_filtration = AttentionFiltration()
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.0)
        self.beta.data.fill_(1.0)

    def l2norm(self, X, dim=-1, eps=1e-8):
        """L2-normalize columns of X"""
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    def forward(self, global_feature, local_feature):
        """
        end-to-end
        :param global_feature: [b, 640]
        :param local_feature: [b, 13, 640]
        :return: [b, 640]
        """
        feature = torch.cat([global_feature.unsqueeze(1), local_feature], 1)
        for module in self.graph:
            feature = module(feature)
        graph_feature = feature[:, 0, :]  # [b, 640]
        graph_feature_origin = graph_feature

        graph_and_local = torch.cat([graph_feature.unsqueeze(1), local_feature], 1)  # [b, 14, 640]
        graph_feature = self.attention_filtration(graph_and_local)

        graph_feature = (graph_feature_origin * self.alpha + graph_feature * self.beta)  # like ResNet

        graph_feature = self.l2norm(graph_feature)
        return graph_feature


class LocalNet(nn.Module):
    def __init__(self, embed_dim=640, dropout=0.5, num_region=13, graph_num=3):
        super(LocalNet, self).__init__()
        self.transformer = MultiFrameIntegrationTransformer(T=num_region)
        self.visual_sa = VisualSA(embed_dim, dropout, num_region)
        self.visual_gr = VisualGR(embed_dim, graph_num)
        self.local_fusion = Combiner(320, 320 * 1, 320 * 2)

    def forward(self, local_feature, mode="out"):
        """
        used for target side
        :param local_feature: [b, 13, 640]
        :param mode
        :return: [b, 640]
        """
        feature_commonality = self.transformer(local_feature)
        feature_sa = self.visual_sa(local_feature)
        feature_difference = self.visual_gr(feature_sa, local_feature)
        feature_out = self.local_fusion(feature_commonality, feature_difference)
        if mode == "all":
            return feature_out, feature_commonality, feature_difference

        return feature_out


class SPIRIT(nn.Module):
    def __init__(self, clip_feature_dim, projection_dim, hidden_dim, clip_model):
        super(SPIRIT, self).__init__()
        self.image_clip = ImageCLIP(clip_model)
        self.text_clip = TextCLIP(clip_model)
        self.combiner = Combiner2(clip_feature_dim, projection_dim, hidden_dim)
        self.visual_attn = LocalNet()

        self.simi_loss = TripletLoss()

    def forward(
        self,
        image=None,
        text=None,
        image_features=None,
        text_features=None,
        target_features=None,
        ref_local_feats=None,
        tar_local_feats=None,
        mode="combine",
    ):
        """
        :param image: reference images
        :param text: reference text
        :param image_features: reference image features
        :param text_features: reference text features
        :param target_features: target image features
        :param ref_local_feats: reference
        :param tar_local_feats: target
        :param mode: combine/image/text/combine_train
        :return: decided by mode
        """
        if mode == "image":
            return self.image_clip(image)
        elif mode == "text":
            return self.text_clip(text)
        elif mode == "combine_train":
            ref_local_attn, _, _ = self.visual_attn(ref_local_feats, mode="all")
            ref_image_features = torch.cat((image_features, ref_local_attn), -1)
            text_features = torch.cat((text_features, text_features), -1)

            predict_features = self.combiner(ref_image_features, text_features)

            tar_local_attn = self.visual_attn(tar_local_feats)
            target_features = torch.cat((target_features, tar_local_attn), -1)
            target_features = F.normalize(target_features, dim=-1)

            return 100 * predict_features @ target_features.T
        elif mode == "local":
            return self.visual_attn(tar_local_feats)
        else:  # combine
            ref_local_attn = self.visual_attn(ref_local_feats)
            ref_image_features = torch.cat((image_features, ref_local_attn), -1)
            text_features = torch.cat((text_features, text_features), -1)
            return self.combiner(ref_image_features, text_features)


if __name__ == "__main__":
    pass
