# Usage

Evaluate:

```shell
MAX_SAMPLES=1 \
NUM_WORKERS=10 \
MODEL_TYPE='gpt4roi' \
CKPT_PATH='/home/t-yutonglin/xiaoke/GPT4RoI/GPT4RoI-7B' \
python -m src.eval \
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
eval_data='[vg-densecap-region_descriptions]'
```


```shell
DATASET_LS=(
vg-densecap-local
refcocog-google
refcoco-unc-split_testA
refcoco-unc-split_testB
refcoco+-unc-split_testA
refcoco+-unc-split_testB
)
for DATASET in ${DATASET_LS[@]}; do
    NUM_WORKERS=4 \
    MODEL_TYPE='gpt4roi' \
    CKPT_PATH='/mnt/blob/weights/GPT4RoI-7B' \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$(nvidia-smi -L | wc -l) \
    -m src.eval \
    eval_data=$DATASET
done
```