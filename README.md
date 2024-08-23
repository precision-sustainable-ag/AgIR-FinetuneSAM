# AgIR-FinetuneSAM

## Overview

`AgIR-FinetuneSAM` is a repository designed to finetune or train models using the `SAM-HQ` framework, developed by the SysCV team. This repository extends the capabilities of `SAM-HQ` for automatic annotation of SemiField and Field images from the AgIR (Agricultural Image Repository).

## Requirements

- Python >= 3.8
- CUDA >= 11.0 (optional but recommended for GPU acceleration)
- All dependencies are listed in the `requirements.txt` file.


## Execute

To execute the training script with distributed training on 8 GPUs, use the following command:

```bash
python torchrun --nproc_per_node=8 train.py
```

## Acknowledgments

- Special thanks to the **SysCV team** for developing the [SAM-HQ](https://github.com/SysCV/sam-hq) repository.
