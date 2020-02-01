# math-recognition

## Installation

-   `conda env -f environment.yml`
-   `conda activate math`
-   `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
-   Install apex

    -   `cd ..`
    -   `git clone https://github.com/NVIDIA/apex`
    -   `cd apex`
    -   `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

-   `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
-   `sudo apt-get install git-lfs`
-   `git lfs install`
-   `git lfs pull`

## Update Conda Env

-   conda deactivate
-   conda env update -f environment.yml
-   conda activate math

## Preprocessing

-   `unzip crohme-2019-unofficial-processed.zip`
-   `sudo tar -xf ./train.tgz -C ./`
-   `sudo tar -xf ./val.tgz -C ./`

-   `sudo python create-iid-dataset.py`

## Run

-   train the model: `allennlp train config.json -s ./logs --include-package math_recognition`
-   evaluate the model

    -   train set: `allennlp evaluate --cuda-device 0 --include-package math_recognition ./logs/model.tar.gz crohme-train/train.csv`
    -   val set: `allennlp evaluate --cuda-device 0 --include-package math_recognition ./logs/model.tar.gz crohme-train/val.csv`

-   predict

    -   train set: `allennlp predict --output-file ./out.txt --batch-size 64 --cuda-device 0 --use-dataset-reader --predictor CROHME --include-package math_recognition --silent ./logs/model.tar.gz crohme-train/train.csv`
    -   val set: `allennlp predict --output-file ./out.txt --batch-size 64 --cuda-device 0 --use-dataset-reader --predictor CROHME --include-package math_recognition --silent ./logs/model.tar.gz crohme-train/val.csv`

-   view predictions
    -   `head -20 out.txt`

## Results

-   train

    -   loss: 0.8128
    -   exprate: 0.6836

-   val
    -   loss: 1.6714
    -   exprate: 0.3663

## Author

Bilal Khan (bilal.software)
