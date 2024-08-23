# AgIR-FinetuneSAM

## Overview

`AgIR-FinetuneSAM` is a repository designed to finetune or train models using the `SAM-HQ` framework, developed by the SysCV team. This repository extends the capabilities of `SAM-HQ` for automatic annotation of SemiField and Field images from the AgIR (Agricultural Image Repository).

## Requirements

- Python >= 3.8
- CUDA >= 11.0 (optional but recommended for GPU acceleration)
- All dependencies are listed in the `requirements.txt` file.

## Installation

### 1. Clone the Repository (with Submodules)

To properly set up the repository and include the `SAM-HQ` submodule, run the following command:

```bash
git clone --recurse-submodules https://github.com/YOUR_ORG/AgIR-FinetuneSAM.git
cd AgIR-FinetuneSAM
```

> **Note**: If you already cloned the repository without the `--recurse-submodules` flag, you can manually initialize and update the submodule:
```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- **Submodule Not Cloned**: If the submodule did not clone correctly, ensure you used the `--recurse-submodules` flag when cloning, or manually initialize the submodule using:
  ```bash
  git submodule update --init --recursive
  ```

## Acknowledgments

- Special thanks to the **SysCV team** for developing the [SAM-HQ](https://github.com/SysCV/sam-hq) repository.
