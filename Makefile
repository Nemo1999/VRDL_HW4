.PHONY: install download_data make_trianing_lr

install: 
	@echo "Installing Dependencies..."
	pip install -r requirements.txt
	cd KAIR_Repo ; pip install -r requirement.txt

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
	@echo "Start training model... "
	@echo "Checkpoints and logs will be saved at KAIR_Repo/superresolution"
	cd KAIR_Repo; python main_train_psnr.py --opt options/train_msrresnet_psnr.json
	

reproduce: 
	rm -rf KAIR_Repo/superresolution/models
	mkdir -p KAIR_Repo/superresolution/models
	@gdown "https://drive.google.com/uc?export=download&id=1sVyh0qiXb16j5Ave_8r2_pbApunes09q" -O "KAIR_Repo/superresolution/models/95000_G.pth"
	@gdown "https://drive.google.com/uc?export=download&id=1PwKxXt9sSAHRJN6A3NbewtbcOYbye8Qd" -O "KAIR_Repo/superresolution/models/95000_E.pth"
	cd KAIR_Repo; python main_test_psnr.py --opt options/test_msrresnet_psnr.json 
	@echo "Done!!!"
	@echo "The resulting 3x HR images are stored in VRDL_HW4/KAIR_Repo/superresolution/images/evaluation{checkpoints iterations}/"