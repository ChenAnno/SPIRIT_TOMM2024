import os
import clip
import numpy as np
import torch
import torch.distributed as dist
from argparse import ArgumentParser
from torch import optim, nn
from torch.utils.data import DataLoader

from dataloader.dataset import targetpad_transform
from dataloader.fashion200k_patch import Fashion200kDataset, Fashion200kTestDataset, Fashion200kTestQueryDataset
from utils.utils import extract_index_features, AverageMeter, setup_seed
from models.model import SPIRIT

setup_seed(42)


def train_200k(
    projection_dim: int,
    hidden_dim: int,
    num_epochs: int,
    clip_model_name: str,
    combiner_lr: float,
    batch_size: int,
    clip_bs: int,
    dataset_name: str,
    **kwargs
):
    """
    Train the Combiner on the Fashion200k dataset keeping frozed the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    global best_r_average

    device = kwargs['device']
    local_rank = kwargs['local_rank']
    train_dress_types, val_dress_types = ['all'], ['all']

    """Define the model"""
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    saved_state_dict = torch.load("ckpt/tuned_{}.pt".format(dataset_name), map_location=device)
    clip_model.load_state_dict(saved_state_dict["CLIP"])
    clip_model.eval().float()

    input_dim = kwargs['input_dim']
    feature_dim = kwargs['feature_dim']
    model = SPIRIT(feature_dim, projection_dim, hidden_dim, clip_model).to(device, non_blocking=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    """Define the training dataset"""
    preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
    relative_train_dataset = Fashion200kDataset(split='train', img_transform=preprocess)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset)
    train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
    )

    """Define the validation datasets and extract the validation index features for each dress_type"""
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []
    index_local_list = []
    for _, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[_] = dress_type
        relative_val_dataset = Fashion200kTestQueryDataset(split='val', img_transform=preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = Fashion200kTestDataset(split='val', img_transform=preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])
        index_local_list.append(index_features_and_names[2])

    """Define the optimizer, the loss and the grad scaler"""
    optimizer = optim.Adam(model.parameters(), lr=combiner_lr)
    cross_entropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * len(train_loader))

    """Train the model, also including validation to witness the best results"""
    best_r_average = 0
    print("Begin to train")
    for epoch in range(num_epochs):
        losses = AverageMeter()
        model.train()
        for idx, (reference_images, target_images, captions, _, _, _, reference_feats, target_feats) in enumerate(train_loader):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            reference_feats = reference_feats.to(device, non_blocking=True)
            target_feats = target_feats.to(device, non_blocking=True)
            text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)

            """Extract the features with CLIP"""
            with torch.no_grad():
                reference_images_list = torch.split(reference_images, clip_bs)
                reference_image_features = torch.vstack(
                    [model(image=mini_batch, mode="image").float() for mini_batch in reference_images_list])
                target_images_list = torch.split(target_images, clip_bs)
                target_image_features = torch.vstack(
                    [model(image=mini_batch, mode="image").float() for mini_batch in target_images_list])

                text_inputs_list = torch.split(text_inputs, clip_bs)
                text_features = torch.vstack(
                    [model(text=mini_batch, mode="text").float() for mini_batch in text_inputs_list])

            """Compute the logits and the loss"""
            with torch.cuda.amp.autocast():
                logits = model(
                    image_features=reference_image_features,
                    text_features=text_features,
                    target_features=target_image_features,
                    ref_local_feats=reference_feats,
                    tar_local_feats=target_feats,
                    mode="combine_train"
                )
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = cross_entropy_criterion(logits, ground_truth)
            losses.update(loss.detach().cpu().item())

            """Backpropagation and update the weights"""
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if local_rank == 0:
                if idx % kwargs["print_frequency"] == 0 or idx == len(train_loader) - 1:
                    print(
                        "Train Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            epoch, idx, len(train_loader), loss=losses
                        )
                    )
                if idx == len(train_loader) - 1:
                    checkpoint_name = f"ckpt/200k-e{epoch}.pt"
                    torch.save(model.module.state_dict(), checkpoint_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='fashion200k', type=str, help="'CIRR' or 'fashionIQ' or 'shoes")
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=4e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--clip-bs", default=4, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--print-frequency", default=100, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    args.world_size = world_size
    args.rank = rank
    args.dist_url = dist_url
    print("=> world size:", world_size)
    print("=> rank:", rank)
    print("=> dist_url:", dist_url)
    print("=> local_rank:", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group(backend="nccl", init_method=dist_url, rank=rank, world_size=world_size)

    training_hyper_params = {
        "input_dim": args.input_dim,
        "feature_dim": args.feature_dim,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "print_frequency": args.print_frequency,
        "device": device,
        "dataset_name": args.dataset.lower(),
        "local_rank": args.local_rank
    }

    train_200k(**training_hyper_params)
