from pathlib import Path

import torch
import tqdm

from model import VQVAE


def compute_logits(
    model: VQVAE,
    dataloader: iter,
    device_model: torch.device = "cpu",
    device_output: torch.device = "cpu",
):
    logits = []
    for images, _ in tqdm.tqdm(dataloader, total=len(dataloader)):
        images = images.to(device_model)
        model(images)
        logits.append(
            torch.argmin(model.codebook.distances[0], dim=-1).to(device_output)
        )

    return torch.cat(logits, dim=0)
