# Transformer and pointer-generator transformer models for the morphological inflection task

## Submission - NYUCUBoulder, Task 0 and Task 2

Run pointer-generator transformer on original datatset and multitask training augmented set (for Task 0).
```bash
bash task0-pg-aug-launch.sh
bash task0-pg-trn-launch.sh
```
Run transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) on original datatset and multitask training augmented set (for Task 0).
```bash
bash task0-trm-aug-launch.sh
bash task0-trm-trn-launch.sh

```

Code built on top of the baseline code for Task 0 for the SIGMORPHON 2020 Shared Tasks [ (Vylomova, 2020)](https://github.com/shijie-wu/neural-transducer.git)
Data hallucination augmentation by [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838)
You can also run hard monotonic attention [(Wu and Cotterell, 2019)](https://arxiv.org/abs/1905.06319).

## Dependencies

- python 3
- pytorch==1.4
- numpy
- tqdm
- fire


## Install

```bash
make
```
