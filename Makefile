# Encodings
latentdim = 1
numembeddings = 1024

# Training
numworkers=1

# Checkpoint
checkpoint=model.ckpt

model: $(checkpoint)
model.ckpt: train.py
	python train.py --model=VQVAE \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --num-workers=$(numworkers)

logits: logits.pt
logits.pt: logits.pt $(checkpoint)
	python train.py --model=VQVAE \
	    --resume=$(checkpoint) \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --num-workers=$(numworkers)
