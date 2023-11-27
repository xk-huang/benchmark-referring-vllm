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
- PVIT https://github.com/PVIT-official/PVIT
    - has weight
        - base weight: llama7b + regionclip
- Shikra https://github.com/shikras/shikra
    - has weight
        - base weight: llama7b

- Kosmos2 https://github.com/microsoft/unilm/tree/master/kosmos-2
    - has weight: itself
    - no code for REG
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

## Evaluate

See [docs/USAGE.md](docs/USAGE.md]).

## Performance

(11/27/23) GPT4RoI: https://github.com/jshilong/GPT4RoI. Weight:  GPT4RoI-7B-delta-V0 https://huggingface.co/shilongz/GPT4RoI-7B-delta-V0/tree/main.

| Model  | Dataset                                              | CIDEr-D | METEOR | SPICE | ROUGE | NounRecall | VerbRecall | NounRecall(Fuzzy) | VerbRecall(Fuzzy) |  
| ------ | ---------------------------------------------------- | ------- | ------ | ----- | ----- | ---------- | ---------- | ----------------- | ----------------- |  
| GPT4RoI| infer-visual_genome-densecap-local-densecap-test.json| 1.1225  | 0.1639 | 0.3028| 0.3241| 0.3913     | 0.04       | 0.6294            | 0.0658            |  
| GPT4RoI| infer-refcoco-refcocog-google-validation.json        | 1.0795  | 0.231  | 0.3147| 0.4769| 0.3904     | 0.1312     | 0.6228            | 0.2142            |  
| GPT4RoI| infer-refcoco-refcoco+-unc-testA.json                | 0.2649  | 0.2736 | 0.2224| 0.4419| 0.2512     | 0.0909     | 0.5689            | 0.2066            |  
| GPT4RoI| infer-refcoco-refcoco+-unc-testB.json                | 0.4412  | 0.2347 | 0.1927| 0.4162| 0.2338     | 0.0339     | 0.4873            | 0.0856            |  
| GPT4RoI| infer-refcoco-refcoco-unc-testA.json                 | 0.2527  | 0.2553 | 0.1913| 0.4273| 0.2474     | 0.0738     | 0.5645            | 0.1911            |  
| GPT4RoI| infer-refcoco-refcoco-unc-testB.json                 | 0.5341  | 0.2565 | 0.1957| 0.4528| 0.2565     | 0.0315     | 0.499             | 0.0778            |  

