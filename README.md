# benchmark-referring-vllm

We benchmark VLLM for referring image captioning. From paper "[Segment and Caption Anything](https://github.com/xk-huang/segment-caption-anything)".

## Current Models

See [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

## Evaluation

See [docs/ENV.md](docs/ENV.md) and [docs/USAGE.md](docs/USAGE.md).

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

