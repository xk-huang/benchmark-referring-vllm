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
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} --name gpt4roi nvcr.io/nvidia/pytorch:22.10-py3 bash
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


## Kosmos-2

```shell
# In host
alias=`whoami | cut -d'.' -f2`
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} --name kosmos-2 nvcr.io/nvidia/pytorch:22.10-py3 bash
docker exec -it kosmos-2 bash

# In docker image
set -e




```