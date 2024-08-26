# AgIR-FinetuneSAM

## Overview

`AgIR-FinetuneSAM` is a repository designed to finetune or train models using the `SAM-HQ` framework, developed by the SysCV team. This repository extends the capabilities of `SAM-HQ` for automatic annotation of SemiField and Field images from the AgIR (Agricultural Image Repository).

## Requirements

Tested on:
- Python 3.11
- torch 2.4.0
- torchvision 0.19.0

To set up the environment and install `SAM-HQ`, follow the [instructions](https://github.com/SysCV/sam-hq?tab=readme-ov-file#example-conda-environment-setup).

## Setup

1. Place the data folder in the "data" directory.
2. Set up the experiment in the "conf/experiments/" directory.

## Execution

To execute the training script with 3 GPUs, use the following command:

```bash
torchrun --nproc_per_node=3 train.py
```

## Acknowledgments

Special thanks to the **SysCV team** for developing the [SAM-HQ](https://github.com/SysCV/sam-hq) repository.