.PHONY: install download_data make_trianing_lr

install: 
	@echo "Installing Dependencies..."
	pip install -r requirements.txt

download_data:
	@echo "Downloading Data..."
	rm -rf data
	mkdir -p data
	gdown https://drive.google.com/uc?id=1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O "data/dataset.zip"
	unzip -q "data/dataset.zip" -d "data/"
	rm "data/dataset.zip"
	mv "data/testing_lr_images/testing_lr_images" "data/testing_lr/"
	mv "data/training_hr_images/training_hr_images" "data/training_hr/"
	rm -rf data/training_hr_images data/testing_lr_images

make_trianing_lr: 
	@echo "Making Training LR Images..."
	rm -rf data/training_lr
	mkdir -p data/training_lr
	python make_training_lr.py

train: 
	cd SwinIR ; python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--master_port=1234 \
		main_train_psnr.py \
		--opt options/swinir/train_swinir_sr_classical.json  \
		--dist True