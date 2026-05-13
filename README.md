# DID

![Generation Process](steps_generation.gif)

This repo contains an official PyTorch implementation for the ICLR'26 paper [Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes](https://openreview.net/forum?id=VbvXjs5f72) by Fangyu Ding, Ding Ding, Sijin Chen, Kaibo Wang, Peng Xu, Zijin Feng, Haoli Bai, Kai Han, Youliang Yan, Binhang Yuan, Jiacheng Sun. 

This work first derives and implements the insertion-based discrete diffusion for text generation, which is mask-free, FLOPs-efficient, variable-length, and with special optimizations for the fixed-length setting.

Only code is available now, the checkpoints can be retrained with the code.

## Dependencies

```shell
# python 3.12, cuda 11.8, torch 2.6, flash-attn 2.7.3
pip install torch-2.6.0+cu118-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```


## Acknowledgements

This repository is built upon [RADD](https://github.com/ML-GSAI/RADD) and [SMDM](https://github.com/ML-GSAI/SMDM). The training datasets are [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), [Stories](https://huggingface.co/datasets/dhruveshpatel/tiny_roc_stories), and SlimPajama.

## Citation

```
@inproceedings{
ding2026beyond,
title={Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes},
author={Fangyu Ding and Ding Ding and Sijin Chen and Kaibo Wang and Peng Xu and Zijin Feng and Haoli Bai and Kai Han and Youliang Yan and Binhang Yuan and Jiacheng Sun},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=VbvXjs5f72}
}
```

