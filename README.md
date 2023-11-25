# benchmark-referring-vllm

We benchmark VLLM for referring image captioning. From paper "[Segment and Caption Anything](https://github.com/xk-huang/segment-caption-anything)".

## Current Models


Baseline: GRiT, for performance check here: https://github.com/xk-huang/Promptable-GRiT

as of 11/25/2023
- GPT4RoI https://github.com/jshilong/GPT4RoI
    - has weight
        - base weight: llama7b
        - replace https://huggingface.co/decapoda-research/llama-7b-hf with https://huggingface.co/baffo32/decapoda-research-llama-7B-hf (or https://huggingface.co/jeffwan/llama-7b-hf/tree/main)
    - report VG in V2 paper: 146; with GRiT
    - prompt: "### Question: Can you give a description of the region mentioned by \<region\> ### Answer:"
- Kosmos2
    - has weight: itself
- Shikra https://github.com/shikras/shikra
    - has weight
        - base weight: llama7b
- PVIT https://github.com/PVIT-official/PVIT
    - has weight
        - base weight: llama7b + regionclip

- GroundingLLM: https://github.com/mbzuai-oryx/groundingLMM 
    - (no weight)
    - compared with GRiT and GPT4ROI, on  refcocog and VG
    - models: Kosmos-2, GPT4RoI, GRIT
    - prompt: "Can you provide a detailed description of the region \<bbox\>"
- All seeing: https://github.com/OpenGVLab/all-seeing
    - no weight
- Ferret: https://github.com/apple/ml-ferret
    - weight
    - In ICLR 24 sub, compared with more models: https://openreview.net/forum?id=2msbbX3ydD
- OPT_Questioner https://github.com/johncaged/OPT_Questioner
    - no weight