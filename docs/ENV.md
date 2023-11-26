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