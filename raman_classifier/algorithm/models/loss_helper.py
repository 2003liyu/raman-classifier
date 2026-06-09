import torch
import torch.nn.functional as F


def compute_dual_path_loss(model, output_tuple, labels, criterion):
    logits_orig, logits_aug, feats_orig, t_logits, lam, index = output_tuple

    # 原图 CE
    loss_orig = criterion(logits_orig, labels)

    # Mixup CE
    log_probs_aug = F.log_softmax(logits_aug, dim=1)
    loss_aug = -torch.mean(
        lam * log_probs_aug.gather(1, labels.unsqueeze(1)).squeeze() +
        (1 - lam) * log_probs_aug.gather(1, labels[index].unsqueeze(1)).squeeze()
    )

    # KD loss（只对 original batch）
    T = model.kd_temperature
    alpha = model.kd_alpha

    kd_loss = F.kl_div(
        F.log_softmax(logits_orig / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean"
    ) * T * T

    return loss_orig + 0.5 * loss_aug + alpha * kd_loss
