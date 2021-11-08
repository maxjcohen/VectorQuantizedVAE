# Encodings
latentdim = 1
numembeddings = 1024

# Training
numworkers=1

# Checkpoint
checkpoint=VQVAE_C_256_N_1_M_1024_D_32/model.ckpt-250000.pt

all: model.ckpt-0.pt
model.ckpt-0.pt: train.py
	python train.py --model=VQVAE \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --num-workers=$(numworkers)

logits: logits.pt
logits.pt: train.py
	python train.py --model=VQVAE \
	    --resume=$(checkpoint) \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --num-workers=$(numworkers)
