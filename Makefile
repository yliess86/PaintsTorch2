FEATURES=32
BATCHSIZE=8
MODEL=models/paintstorch2_guide_40/checkpoint_39.pth
GUIDE=--guide
BN=
CURRICULUM=

build:
	sudo docker build -t yliess86/paintstorch2:latest .
	sudo docker push yliess86/paintstorch2:latest

train:
	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		${GUIDE} ${BN} ${CURRICULUM}

exp:
	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		--guide --curriculum

	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		--guide

	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE}


	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		--guide --bn --curriculum

	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		--guide --bn
	
	python3 -m kubeflow \
		--n_gpu 4 \
		--batch_size ${BATCHSIZE} \
		--features ${FEATURES} \
		--num_workers ${BATCHSIZE} \
		--bn

test:
	python -m evaluation.benchmark \
		--features ${FEATURES} \
		--batch_size ${BATCHSIZE} \
		--num_workers ${BATCHSIZE} \
		--dataset dataset \
		--model ${MODEL} \
		${BN}

	python -m evaluation.data \
		--dataset dataset

convert:
	python -m evaluation.export \
		--features ${FEATURES} \
		--model ${MODEL} \
		--save docs/resources/paintstorch.onnx

	rm -rf docs/resources/paintstorch.onnx docs/resources/paintstorch.pb