# Usage

Evaluate:

```shell
MAX_SAMPLES=1 \
NUM_WORKERS=10 \
MODEL_TYPE='gpt4roi' \
CKPT_PATH='/home/t-yutonglin/xiaoke/GPT4RoI/GPT4RoI-7B' \
python -m src.eval \
train_data='[vg-densecap-region_descriptions]' \
eval_data='[vg-densecap-region_descriptions]'


MAX_SAMPLES=1 \
NUM_WORKERS=4 \
MODEL_TYPE='gpt4roi' \
CKPT_PATH='/home/t-yutonglin/xiaoke/GPT4RoI/GPT4RoI-7B' \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
-m src.eval \
train_data='[vg-densecap-region_descriptions]' \
eval_data='[vg-densecap-region_descriptions]'
```