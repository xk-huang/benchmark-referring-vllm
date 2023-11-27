# ENV

## GPT4ROI

warning: need to delete the check in transformers per https://github.com/jshilong/GPT4RoI/issues/14#issuecomment-1674193283.
- File "/opt/conda/lib/python3.8/site-packages/transformers/generation/utils.py", line 1271,

```shell
git submodule add https://github.com/jshilong/GPT4RoI.git third_party/GPT4RoI
git submodule update --init --recursive
touch third_party/GPT4RoI/gpt4roi/__init__.py
```


```shell
alias=`whoami | cut -d'.' -f2`
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} -w `pwd` --name gpt4roi nvcr.io/nvidia/pytorch:22.10-py3 bash
docker exec -it gpt4roi bash

# cd /home/t-yutonglin/xiaoke/GPT4RoI/
# cd /home/t-yutonglin/xiaoke/benchmark-referring-vllm/

# In docker image
. amlt_configs/setup-gpt4roi.sh
```

```shell
# In host
cd third_party/GPT4RoI/
git lfs install
git clone https://huggingface.co/decapoda-research/llama-7b-hf ./llama-7b 
git lfs install
git clone https://huggingface.co/shilongz/GPT4RoI-7B-delta-V0 ./GPT4RoI-7B-delta
cd -
python -m scripts.apply_delta \
    --base /home/t-yutonglin/xiaoke/decapoda-research-llama-7B-hf \
    --target ./GPT4RoI-7B \
    --delta ./GPT4RoI-7B-delta
```

## PVIT

```shell
https://github.com/PVIT-official/PVIT.git
```


```shell
alias=`whoami | cut -d'.' -f2`
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} -w `pwd` --name pvit nvcr.io/nvidia/pytorch:22.10-py3 bash
docker exec -it pvit bash

# cd /home/t-yutonglin/xiaoke/PVIT
```

```shell
cp requirements.txt requirements.txt.bak
sed -i '3d' requirements.txt.bak
sed -i '3d' requirements.txt.bak
sed -i '3d' requirements.txt.bak
pip install -r requirements.txt.bak
git clone https://github.com/microsoft/RegionCLIP.git
pip install -e RegionCLIP

# NOTE: dependency hell.
pip install pydantic==1.10.10

mkdir model_weights
# pip install gdown
# gdown "https://drive.google.com/uc?id=1-6u-55s0izj1nbuv7yVOSo0jbdjcyHd_"
# https://drive.usercontent.google.com/download?id=1-6u-55s0izj1nbuv7yVOSo0jbdjcyHd_&export=download&authuser=0&confirm=t&uuid=7624d146-63e9-477c-9b10-f224c40b77df&at=APZUnTXH3z07sciUoImamil0-gW7:1700989428097
# git lfs install
# git clone https://huggingface.co/PVIT/pvit model_weights/pvit-delta


BASE_MODEL=../decapoda-research-llama-7B-hf TARGET_MODEL=model_weights/pvit DELTA=model_weights/pvit-delta ./scripts/delta_apply.sh



MODEL_PATH=model_weights/pvit CONTROLLER_PORT=39996 WORKER_PORT=40004 ./scripts/model_up.sh

MODEL_ADDR=http://0.0.0.0:40004 ./scripts/run_cli.sh

```


```shell
git submodule add https://github.com/PVIT-official/PVIT.git third_party/PVIT


. amlt_configs/setup-pvit.sh
```


## Evaluate metrics with `vdtk`

### Install `vdtk`

Support CLIP computation with images encoded by base64.

https://github.com/xk-huang/vdtk/tree/dev

- data (e.g., jar files): https://huggingface.co/xk-huang/vdtk-data

Install with external data:

```shell
ORIGINAL_DIR="$(pwd)"
REPO_DIR=/tmp/vdtk
git clone --recursive https://github.com/xk-huang/vdtk.git $REPO_DIR -b dev
cd $REPO_DIR
git submodule update --init --recursive

apt-get update
sudo apt-get update
apt-get install git-lfs
sudo apt-get install git-lfs

git lfs install
git clone https://huggingface.co/xk-huang/vdtk-data
# git submodule init && git submodule update

rsync -avP ./vdtk-data/vdtk .
rm -rf vdtk-data

pip install --upgrade pip
pip install -e . POT==0.9.0  # POT=0.9.1 will take up all the memory with tf backend
pip install tensorflow==2.12.1  # Just fix one version of tf
pip install levenshtein==0.21.1
pip install openpyxl==3.1.2

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cd "$ORIGINAL_DIR"
```

Potential Problems:

- About Tensorflow: TF does not support CUDA 12 now (08/15/23). So we use `nvcr.io/nvidia/pytorch:22.12-py3` which contains CUDA 11.8.
- Encoding in docker image: `import locale;locale.getpreferredencoding()` is `ANSI_X3.4-1968` rather than `UTF-8` which causes error in file writing.
  - change `vdtk/metrics/tokenizer/ptbtokenizer.py:73`: `tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")`


### The format of input prediction json file

```json
[
    {
        "_id": 0,
        "split": "inference",
        "references": [
            "red and yellow",
            "red shirt guy",
            "red and yellow uniform"
        ],
        "candidates": [
            "a man wearing a red and white shirt"
        ],
        "metadata": {
            "metadata_input_boxes": [
                0,
                95,
                113,
                419
            ],
            "metadata_image_id": 266240,
            "metadata_region_id": 27287
        },
        "logits": {
            "iou_scores": [
                0.89990234375,
                0.994140625,
                0.99365234375
            ]
        }
    }
]
```