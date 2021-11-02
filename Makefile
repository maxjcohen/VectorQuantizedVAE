# Encodings
latentdim = 1
numembeddings = 16

# Encoder
channels = 32

# Training
batchsize = 16
trainingsteps = 5

all: model.ckpt-0.pt
model.ckpt-0.pt: train.py
	python train.py --model=VQVAE \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --channels=$(channels) \
	    --batch-size=$(batchsize) \
	    --num-training-steps=$(trainingsteps)
