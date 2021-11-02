from pathlib import Path

import torch

from model import VQVAE


def compute_logits(
    model: VQVAE, dataloader: iter, device: torch.device = "cpu"
):
    logits = []
    for images, _ in dataloader:
        images = images.to(device)
        model(images)
        logits.append(model.codebook.distances.squeeze())

    return torch.cat(logits, dim=0)
