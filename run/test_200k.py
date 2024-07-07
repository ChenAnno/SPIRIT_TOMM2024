from argparse import ArgumentParser
import clip
import numpy as np
import torch
from models.model import SPIRIT
from operator import itemgetter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from statistics import mean
from tqdm import tqdm

from dataloader.dataset import targetpad_transform
from utils.utils import extract_index_features, collate_fn
from dataloader.fashion200k_patch import Fashion200kTestDataset, Fashion200kTestQueryDataset


def compute_200k_val_metrics(
    relative_val_dataset,
    clip_model,
    index_features,
    index_local_features,
    index_names,
    model,
    device,
):
    """
    Compute validation metrics on Fashion200k dataset
    :param relative_val_dataset: Fashion200k validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_local_features
    :param index_names: validation index names
    :param model: function which takes as input (image_features, text_features) and outputs the combined features
    :param device
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_200k_val_predictions(
        clip_model, relative_val_dataset, model, index_names, index_features, device
    )

    # Normalize the index features
    predicted_features = predicted_features.cpu()
    model, index_local_features, index_features = model.cpu(), index_local_features.cpu(), index_features.cpu()

    tar_local_attn = model(tar_local_feats=index_local_features, mode="local")
    index_features = torch.cat((index_features, tar_local_attn), -1)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances.cpu(), dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = []
    for sorted_index_name, target_name in tqdm(zip(sorted_index_names, target_names)):
        label = torch.tensor(sorted_index_name == np.repeat(np.array(target_name), len(index_names)))
        labels.append(label)
    labels = torch.stack(labels)

    # Compute the metrics
    recall_at10 = (torch.sum(torch.where(torch.sum(labels[:, :10], dim=1) > 0, 1.0, 0.0)) / len(labels)).item() * 100
    recall_at50 = (torch.sum(torch.where(torch.sum(labels[:, :50], dim=1) > 0, 1.0, 0.0)) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_200k_val_predictions(
    clip_model,
    relative_val_dataset,
    model,
    index_names,
    index_features,
    device,
):
    """
    Compute Fashion200k predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param model: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features [b, dim]
    :param index_names: validation index names
    :param device
    :return: predicted features and target names
    """
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=32,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    name_to_feat = dict(zip(index_names, index_features))

    predicted_features = torch.empty((0, clip_model.visual.output_dim * 2)).to(device, non_blocking=True)
    target_names = []

    for _, ref_names, captions, batch_target_names, _, ref_patch_feats in relative_val_loader:
        text_inputs = clip.tokenize(captions, context_length=77).to(device, non_blocking=True)
        ref_patch_feats = ref_patch_feats.to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*ref_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*ref_names)(name_to_feat))
            reference_image_features = reference_image_features.to(device)
            batch_predicted_features = model(
                image_features=reference_image_features,
                text_features=text_features,
                ref_local_feats=ref_patch_feats,
                mode="combiner",
            )
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='fashion200k', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k" or 'shoes')
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--patch-num", default=13, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'ViT-B-16', 'RN50x4'")
    parser.add_argument("--model-path", type=str, help="Path to the fine-tuned fusion model")
    args = parser.parse_args()

    device = torch.device("cuda")

    """Define the model and the train dataset"""
    clip_model, _ = clip.load(args.dataset, device=device, jit=False)
    clip_model.eval()
    clip_model = clip_model.float()
    model = SPIRIT(args.feature_dim, args.feature_dim * 4, args.feature_dim * 8, clip_model).to(device, non_blocking=True)
    model.load_state_dict(torch.load(args.model_path))

    """Define the test dataset"""
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_whole_features_list = []
    index_names_list = []
    index_local_list = []
    preprocess = targetpad_transform(args.target_ratio, args.input_dim)
    for _, dress_type in enumerate(['all']):
        idx_to_dress_mapping[_] = dress_type
        relative_val_dataset = Fashion200kTestQueryDataset(split='val', img_transform=preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = Fashion200kTestDataset(split='val', img_transform=preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_whole_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])
        index_local_list.append(index_features_and_names[2])

    """Begin to test, compute the recall metrics"""
    recalls_at10, recalls_at50 = [], []
    for relative_val_dataset, index_features, index_names, index_local_feats, _ in zip(
            relative_val_datasets,
            index_whole_features_list,
            index_names_list,
            index_local_list,
            idx_to_dress_mapping,
    ):
        recall_at10, recall_at50 = compute_200k_val_metrics(
            relative_val_dataset,
            clip_model,
            index_features,
            index_local_feats,
            index_names,
            model,
            device,
        )

        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)

    r_10, r_50 = mean(recalls_at10), mean(recalls_at50)
    r_average = (r_10 + r_50) / 2

    print("R@10: ", r_10)
    print("R@50: ", r_50)
    print("Average: ", r_average)
