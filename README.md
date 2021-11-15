# Vector Quantized VAE

## Setup

1. Install requirements and vqvae package
```bash
python3 -m venv venv
. venv/bin/activate
pip install --ugprade pip wheel
pip install torch
pip install git+https://github.com/maxjcohen/VectorQuantizedVAE/@package
```

2. Download model checkpoint
```bash
curl https://cloud.zagouri.org/index.php/s/dpQRnC4ZbFeKwCo/download -o model.ckpt
```

3. Instanticate model and load checkpoint
```python
from vqvae.model import VQVAE

vqvae = VQVAE(channels=256,
              latent_dim=1,
              num_embeddings=1024,
              embedding_dim=32)
checkpoint = torch.load("model.ckpt", map_location=lambda storage, loc: storage)
vqvae.load_state_dict(checkpoint["model"])
```
