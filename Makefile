# Encodings
latentdim = 1
numembeddings = 512

# Encoder
channels = 256

# Training
batchsize = 32
trainingsteps = 100000
numworkers=1

all: model.ckpt-0.pt
model.ckpt-0.pt: train.py
	python train.py --model=VQVAE \
	    --latent-dim=$(latentdim) \
	    --num-embeddings=$(numembeddings) \
	    --channels=$(channels) \
	    --batch-size=$(batchsize) \
	    --num-training-steps=$(trainingsteps) \
	    --num-workers=$(numworkers)
