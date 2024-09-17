
# Fine-Tuning SAM Pipeline

This repository contains scripts for fine-tuning a SAM (Segment Anything Model) model on custom datasets and comparing performance before and after fine-tuning. The scripts are designed to automate the training pipeline and store relevant logs and metrics for detailed performance evaluation.

## Repository Contents

- `FINE_TUNE_SAM_pipeline.py`: This script runs the entire fine-tuning pipeline, including data preparation, model loading, training, and evaluation.
- `fine_tune_OOP.py`: This script implements the fine-tuning pipeline in an object-oriented programming (OOP) paradigm, allowing more flexibility and modularity in running the pipeline.

## Key Features

- Fine-tuning the SAM model with user-defined hyperparameters.
- Automated logging of results, including before-and-after comparisons.
- Support for multiple datasets and customized training options.
- Comparison of the model's performance before and after fine-tuning using metrics such as IoU, accuracy, and loss.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/fine-tune-sam.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Fine-Tuning Pipeline

To start fine-tuning the SAM model, use the main pipeline script:

```bash
python FINE_TUNE_SAM_pipeline.py
```

### Configuration

The pipeline is highly configurable using the `config.yaml` file. You can modify parameters such as:

- Dataset path
- Model hyperparameters (learning rate, batch size, etc.)
- Training epochs
- Evaluation metrics

## Outputs

- **Logs:** Training and evaluation logs are saved in the specified `save_dir`.
- **Reports:** Model performance comparison reports are generated in the `report` folder, organized by date.
- **Model Checkpoints:** Fine-tuned models are saved for future inference or further tuning.


## License

This project is licensed under the MIT License.
