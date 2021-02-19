FEATURES=32
BATCHSIZE=8
MODEL=models/paintstorch2_bn_100/checkpoint_99.pth
GUIDE=--guide
BN=--bn

build:
	sudo docker build -t yliess86/paintstorch2:latest .
	sudo docker push yliess86/paintstorch2:latest

train:
	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		${GUIDE} ${BN}

test:
	python -m evaluation.fid \
		--features ${FEATURES} \
		--batch_size ${BATCHSIZE} \
		--num_workers ${BATCHSIZE} \
		--dataset dataset \
		--model ${MODEL}

	python -m evaluation.perceptual \
		--features ${FEATURES} \
		--batch_size ${BATCHSIZE} \
		--num_workers ${BATCHSIZE} \
		--dataset dataset \
		--model ${MODEL}

	python -m evaluation.data \
		--dataset dataset