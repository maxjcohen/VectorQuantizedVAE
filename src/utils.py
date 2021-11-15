from pathlib import Path

import torch
import tqdm

from vqvae.model import VQVAE


@torch.no_grad()
def compute_logits(
    model: VQVAE,
    dataloader: iter,
    device_model: torch.device = "cpu",
    device_output: torch.device = "cpu",
    indexes: bool = False,
):
    """Returns codebook associated with dataset.

    Iterate through the dataset, storing codebook associated with each batch. If
    `indexes` is set to `True`, only store the `argmin` instead of the logits.

    Parameters
    ==========
    model: VQVAE model
    dataloader: iterator over the dataset.
    device_model: device for storing inputs. Default is `"cpu"`.
    device_output: device for storing the codebooks. Because the size of the tensor can
    be considerably big, it is advised to stick to `"cpu"`. Default is `"cpu"`.
    indexes: if `True`, stores the `argmin` of the logits. Default is `False`.

    Returns
    =======
    Codebooks tensor with shape `(N, W, H, M)` if `indexes` is set to `False`, otherwise
    `(N, W, H)`. `N` is the number of samples in the dataset, `W` and `H` are the
    dimensions of the feature map, `M` is the number of codebooks available.
    """
    logits = []
    for images, _ in tqdm.tqdm(dataloader, total=len(dataloader)):
        images = images.to(device_model)
        model(images)
        logit = model.codebook.distances[0].to(device_output)
        if indexes:
            logit = torch.argmin(logit, dim=-1)
        logits.append(logit)

    return torch.cat(logits, dim=0)
