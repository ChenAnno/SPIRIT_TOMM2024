import clip
import os
from argparse import ArgumentParser
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from dataloader.dataset import CIRRDataset, targetpad_transform
from utils.utils import extract_index_features, AverageMeter, setup_seed
from models.model import SPIRIT
from run.validate_cirr import compute_cirr_val_metrics

setup_seed(42)

def train_cirr(
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
    Train the Combiner on the CIRR dataset keeping frozed the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param: dataset_name: cirr
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    global best_r_average
    device = kwargs['device']
    local_rank = kwargs['local_rank']

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
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
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
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    val_index_features, val_index_names, val_index_patches = extract_index_features(classic_val_dataset, clip_model)

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
        for idx, (reference_images, target_images, captions, reference_feats, target_feats) in enumerate(train_loader):
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

            if idx == len(train_loader) - 1 or idx % kwargs["print_frequency"] == 0:
                if local_rank == 0:
                    print(
                        "Train Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            epoch,
                            idx,
                            len(train_loader),
                            loss=losses,
                        )
                    )
                if idx == len(train_loader) - 1:
                    model.eval()
                    model = model.float()
                    """Compute and log validation metrics for each validation dataset"""
                    results = compute_cirr_val_metrics(
                        relative_val_dataset,
                        clip_model,
                        val_index_features,
                        val_index_patches,
                        val_index_names,
                        model,
                    )
                    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                    r_average = (recall_at5 + group_recall_at1) / 2

                    if local_rank == 0:
                        print("GR@1: ", group_recall_at1, "    GR@2: ", group_recall_at2, "    GR@3: ", group_recall_at3)
                        print("R@1: ", recall_at1, "    R@5: ", recall_at5, "R@10: ", recall_at10, "    R@50: ", recall_at50)
                        if r_average > best_r_average:
                            best_r_average = round(r_average, 5)
                        if local_rank == 0:
                            print("Best Mean Now: ", best_r_average, "*" * 30)
                        # save the checkpoint
                        checkpoint_name = "ckpt/{}-best".format(dataset_name) + ".pth"
                        torch.save(model.module.state_dict(), checkpoint_name)
                    else:
                        if local_rank == 0:
                            print("Mean Now: {:.5f} Best Before: {:.5f} {}".format(r_average, best_r_average, "-" * 20))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='CIRR', type=str, help="name")
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
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

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

    train_cirr(**training_hyper_params)
