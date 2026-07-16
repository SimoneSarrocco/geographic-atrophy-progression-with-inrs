import argparse
import ast
import os

import monai
import albumentations as A
import numpy as np
import torch
from data_utils.prepare_dataset import prepare_dataset_segmentation
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import dice_coeff, hausdorff
from utils.parse import parse_settings
from utils.seed import seed_everything


def add_random_noise(img: torch.Tensor, max_intensity: float = 0.1) -> torch.Tensor:
    intensity = max_intensity * torch.rand(1).to(img.device)
    noise = intensity * torch.randn_like(img)
    return img + noise


def seg_loss_fn(pred, target, pos_weight: float = 19.0, dice_eps: float = 1.0):
    """Weighted BCE + soft Dice on probabilities (the model output is post-sigmoid).

    GA is only ~5% of pixels; plain BCELoss is minimized by predicting all-background, which
    collapses DICE to 0. pos_weight (~ neg/pos ratio) upweights GA in the BCE, and the soft-Dice
    term is imbalance-robust and explicitly penalizes the all-zero solution.
    """
    p = pred.clamp(1e-6, 1.0 - 1e-6)
    bce = -(pos_weight * target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)).mean()
    inter = (pred * target).sum()
    dice = 1.0 - (2.0 * inter + dice_eps) / (pred.sum() + target.sum() + dice_eps)
    return bce + dice


def save_seg_panel(out_path: str, faf2d, gt2d, prob2d, title: str = "", thresh: float = 0.5,
                   writer=None, tag: str = None, step: int = None) -> None:
    """Save a [FAF | GT GA | Pred GA prob | Pred GA (binary) | Change] panel for one image.
    - Pred GA prob is continuous in [0,1] (an all-background collapse shows as near-black with low
      maxGAprob).
    - Pred GA (binary) thresholds the prob at `thresh` so it is directly comparable to the binary GT.
    - Change colour-codes the binarised prediction vs the binary GT: TP=green, FP=red (over-
      segmented), FN=blue (missed), TN=black; its title shows the per-image foreground Dice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    gt_bin = np.asarray(gt2d).squeeze() > 0.5
    prob = np.asarray(prob2d).squeeze()
    pred_bin = prob > thresh
    tp = gt_bin & pred_bin
    fp = pred_bin & ~gt_bin
    fn = gt_bin & ~pred_bin
    denom = int(pred_bin.sum() + gt_bin.sum())
    dice = 1.0 if denom == 0 else float(2.0 * int(tp.sum()) / denom)          # both-empty -> 1.0
    change = np.zeros(gt_bin.shape + (3,), dtype=np.float32)                  # TN stays black
    change[tp] = (0.0, 1.0, 0.0)        # green  = correct GA
    change[fp] = (1.0, 0.0, 0.0)        # red    = false positive (over-segment)
    change[fn] = (0.0, 0.4, 1.0)        # blue   = false negative (missed GA)

    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    ax[0].imshow(np.asarray(faf2d).squeeze(), cmap="gray"); ax[0].set_title("FAF")
    ax[1].imshow(gt_bin, cmap="gray", vmin=0, vmax=1); ax[1].set_title("GT GA")
    ax[2].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[2].set_title("Pred GA prob")
    ax[3].imshow(pred_bin, cmap="gray", vmin=0, vmax=1); ax[3].set_title("Pred GA (>%.2g)" % thresh)
    ax[4].imshow(change); ax[4].set_title("Change  Dice=%.3f" % dice)
    ax[4].legend(handles=[Patch(color=(0, 1, 0), label="TP"), Patch(color=(1, 0, 0), label="FP"),
                          Patch(color=(0, 0.4, 1), label="FN")],
                 loc="lower right", fontsize=6, framealpha=0.4)
    for a in ax:
        a.axis("off")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    # Also push the same panel to TensorBoard's Images tab (if a SummaryWriter is passed).
    if writer is not None and tag is not None:
        try:
            writer.add_figure(tag, fig, global_step=step)
        except Exception:
            pass
    plt.close(fig)


def save_weights(model_save_path: str, model):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    return


def load_weights(model_save_path: str, model, device: torch.device):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data augmentation = EXACTLY the ORIGINAL ImageFlowNet segmentor augmentation (git
    # 62c22af "add segmentor for GA"): ShiftScaleRotate + RandomBrightnessContrast on the TRAIN
    # split only (same params). Applied to image+mask jointly BEFORE normalisation in the dataset
    # (RetinaFafGaSegDataset.__getitem__), so geometry stays consistent and intensity is renormalised
    # afterwards. No augmentations beyond what the original repo implemented.
    train_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])
    transforms_list = [train_transform, None, None]

    train_set, val_set, _, num_image_channel = \
        prepare_dataset_segmentation(config=config, transforms_list=transforms_list)

    # Build the model
    model = torch.nn.Sequential(
        monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=num_image_channel,
            out_channels=1,
            kernel_size=[5, 5, 5, 5],
            filters=[16, 32, 64, 128],
            strides=[1, 1, 1, 1],
            upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)

    # Optional fine-tuning: initialise from a released checkpoint (e.g. the UCSF segmentor) before
    # training on this cohort. Same DynUNet arch, so a strict load works.
    init_from = config.get('init_from', None)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location=device))
        log('Segmentor INITIALISED from %s (fine-tuning).' % init_from, to_console=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate)

    pos_w = float(config.get('seg_pos_weight', 19.0))   # ~ neg/pos ratio for ~5% GA
    viz_every = int(config.get('viz_every', 10))
    best_val_dice = -1.0                                 # select on DICE, NOT BCE (BCE rewards collapse)

    # TensorBoard: scalars (train/val loss + DICE) and the FAF|GT|prob|binary|change panels (Images
    # tab). `tensorboard --logdir <save_folder>/tb`.
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=config.save_folder + 'tb/')

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss = 0
        train_metrics = {
            'dice': [],
            'hausdorff': [],
        }

        model.train()
        for iter_idx, (x_train, seg_true) in enumerate(tqdm(train_set)):
            if 'max_training_samples' in config:
                if iter_idx * config.batch_size > config.max_training_samples:
                    break

            # add_random_noise(x_train) DISABLED (per request): no input perturbation.
            x_train = x_train.float().to(device)
            seg_pred = model(x_train)
            seg_pred = seg_pred.squeeze(1).float().to(device)
            if len(seg_true.shape) == 4:
                seg_true = seg_true.squeeze(1)
            seg_true = seg_true.float().to(device)

            loss = seg_loss_fn(seg_pred, seg_true, pos_weight=pos_w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Visualize a few TRAIN predictions every `viz_every` epochs (GA probability map).
            if epoch_idx % viz_every == 0 and iter_idx == 0:
                for b in range(min(3, seg_pred.shape[0])):
                    save_seg_panel(
                        '%strain_images/epoch%s_sample%d.png' % (config.save_folder, str(epoch_idx).zfill(4), b),
                        x_train[b, 0].detach().cpu().numpy(),
                        seg_true[b].detach().cpu().numpy(),
                        seg_pred[b].detach().cpu().numpy(),
                        title='train ep%d  maxGAprob=%.2f' % (epoch_idx, float(seg_pred[b].max())),
                        writer=tb, tag='train_panels/sample%d' % b, step=epoch_idx)

            seg_true = seg_true.cpu().detach().numpy()
            seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

            for batch_idx in range(seg_true.shape[0]):
                train_metrics['dice'].append(
                    dice_coeff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))
                train_metrics['hausdorff'].append(
                    hausdorff(label_pred=seg_pred[batch_idx, ...],
                              label_true=seg_true[batch_idx, ...]))

        train_loss = train_loss / len(train_set)
        tb.add_scalar('train/loss', train_loss, epoch_idx)
        tb.add_scalar('train/dice', float(np.mean(train_metrics['dice'])), epoch_idx)

        log('Train [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff:  %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, train_loss,
             np.mean(train_metrics['dice']), np.std(train_metrics['dice']) /
             np.sqrt(len(train_metrics['dice'])),
             np.mean(train_metrics['hausdorff']),
             np.std(train_metrics['hausdorff']) /
             np.sqrt(len(train_metrics['hausdorff']))),
            filepath=config.log_dir,
            to_console=False)

        val_loss = 0
        model.eval()
        val_metrics = {
            'dice': [],
            'hausdorff': [],
        }
        with torch.no_grad():
            for _, (x_val, seg_true) in enumerate(val_set):
                x_val = x_val.float().to(device)

                seg_pred = model(x_val)
                seg_pred = seg_pred.squeeze(1).float().to(device)
                if len(seg_true.shape) == 4:
                    seg_true = seg_true.squeeze(1)
                seg_true = seg_true.float().to(device)

                loss = seg_loss_fn(seg_pred, seg_true, pos_weight=pos_w)

                val_loss += loss.item()

                # Visualize a few VAL predictions every `viz_every` epochs (GA probability map).
                if epoch_idx % viz_every == 0 and _ == 0:
                    for b in range(min(3, seg_pred.shape[0])):
                        save_seg_panel(
                            '%sval_images/epoch%s_sample%d.png' % (config.save_folder, str(epoch_idx).zfill(4), b),
                            x_val[b, 0].detach().cpu().numpy(),
                            seg_true[b].detach().cpu().numpy(),
                            seg_pred[b].detach().cpu().numpy(),
                            title='val ep%d  maxGAprob=%.2f' % (epoch_idx, float(seg_pred[b].max())),
                            writer=tb, tag='val_panels/sample%d' % b, step=epoch_idx)

                seg_true = seg_true.cpu().detach().numpy()
                seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

                for batch_idx in range(seg_true.shape[0]):
                    val_metrics['dice'].append(
                        dice_coeff(label_pred=seg_pred[batch_idx, ...],
                                   label_true=seg_true[batch_idx, ...]))
                    val_metrics['hausdorff'].append(
                        hausdorff(label_pred=seg_pred[batch_idx, ...],
                                  label_true=seg_true[batch_idx, ...]))

        val_loss = val_loss / len(val_set)

        log('Validation [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff: %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, val_loss,
             np.mean(val_metrics['dice']),
             np.std(val_metrics['dice']) / np.sqrt(len(val_metrics['dice'])),
             np.mean(val_metrics['hausdorff']),
             np.std(val_metrics['hausdorff']) / np.sqrt(
                 len(val_metrics['hausdorff']))),
            filepath=config.log_dir,
            to_console=False)

        val_dice_mean = float(np.mean(val_metrics['dice']))
        tb.add_scalar('val/loss', val_loss, epoch_idx)
        tb.add_scalar('val/dice', val_dice_mean, epoch_idx)
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            save_weights(config.model_save_path, model)
            log('Model weights saved (best val dice %.3f).' % val_dice_mean,
                filepath=config.log_dir,
                to_console=False)

    return


def test(config: AttributeHashmap):
    device = torch.device('cpu')
    _, _, test_set, num_image_channel = \
        prepare_dataset_segmentation(config=config)

    # Build the model
    model = torch.nn.Sequential(
        monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=num_image_channel,
            out_channels=1,
            kernel_size=[5, 5, 5, 5],
            filters=[16, 32, 64, 128],
            strides=[1, 1, 1, 1],
            upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)

    load_weights(config.model_save_path, model, device=device)
    log('Model weights successfully loaded.',
        to_console=True)

    loss_fn = torch.nn.BCELoss()

    test_loss = 0
    test_metrics = {
        'dice': [],
        'hausdorff': [],
    }
    model.eval()

    with torch.no_grad():
        for _, (x_test, seg_true) in enumerate(test_set):
            x_test = x_test.float().to(device)
            seg_pred = model(x_test)
            seg_pred = seg_pred.squeeze(1).type(
                torch.FloatTensor).to(device)
            if len(seg_true.shape) == 4:
                seg_true = seg_true.squeeze(1)
            seg_true = seg_true.float().to(device)

            loss = loss_fn(seg_pred, seg_true)

            seg_true = seg_true.cpu().detach().numpy()
            seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

            for batch_idx in range(seg_true.shape[0]):
                test_metrics['dice'].append(
                    dice_coeff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))
                test_metrics['hausdorff'].append(
                    hausdorff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))

            test_loss += loss.item()

    test_loss = test_loss / len(test_set)

    log('Test loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff: %.3f \u00B1 %.3f.'
        % (test_loss, np.mean(test_metrics['dice']),
           np.std(test_metrics['dice']) / np.sqrt(len(test_metrics['dice'])),
           np.mean(test_metrics['hausdorff']),
           np.std(test_metrics['hausdorff']) / np.sqrt(
               len(test_metrics['hausdorff']))),
        filepath=config.log_dir,
        to_console=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=None, type=int)

    parser.add_argument('--dataset-name', default='retina_faf_ga', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    parser.add_argument('--crop-size', default=620, type=int,
                        help="centered crop of native 768 before resize. 620 = registration-frame crop "
                             "(default); 768 = NO crop (pure resize 768->target, e.g. the 256 track).")
    parser.add_argument('--init-from', default=None, type=str,
                        help="optional checkpoint to INITIALISE the segmentor from before training on "
                             "this cohort (e.g. the released $ROOT/checkpoints/segment_retinaUCSF_seed1.pty) "
                             "-> fine-tuning instead of from-scratch. Same DynUNet arch required.")
    # parser.add_argument('--image-folder', default='UCSF_images_final_512x512', type=str)
    # parser.add_argument('--mask-folder', default='UCSF_masks_final_512x512', type=str)
    # parser.add_argument('--dataset-path', default='$ROOT/data/retina_ucsf/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty', type=str)

    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=1024, type=int)  # this also extends dataset
    parser.add_argument('--seg-pos-weight', default=19.0, type=float)  # GA upweight in BCE (~neg/pos ratio)
    parser.add_argument('--viz-every', default=10, type=int)           # save GA-prob panels every N epochs

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, segmentor=True, log_settings=config.mode == 'train', run_count=config.run_count)

    assert config.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if config.mode == 'train':
        train(config=config)
        test(config=config)
    elif config.mode == 'test':
        test(config=config)
