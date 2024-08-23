# AgIR-FinetuneSAM

## Overview

`AgIR-FinetuneSAM` is a repository designed to finetune or train models using the `SAM-HQ` framework, developed by the SysCV team. This repository builds upon the capabilities of `SAM-HQ` to train models for automatic annotation of SemiField and Field images, datasets in the AgIR (Agricultural Image Repository).

  
## Requirements

- Python >= 3.8
- CUDA >= 11.0 (optional but recommended for GPU acceleration)
- Dependencies are listed in the `requirements.txt` file.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YOUR_ORG/AgIR-FinetuneSAM.git
   cd AgIR-FinetuneSAM
   ```

2. **Add the SAM-HQ submodule**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Acknowledgments

- **SysCV team** for developing the [SAM-HQ](https://github.com/SysCV/sam-hq) repository.
