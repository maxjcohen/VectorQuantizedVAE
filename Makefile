# Encodings
latentdim = 1
numembeddings = 1024

# Training
numworkers=1

all: model.ckpt-0.pt
model.ckpt-0.pt: train.py
	python train.py --model=VQVAE \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --num-workers=$(numworkers)
