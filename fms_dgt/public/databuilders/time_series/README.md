# Time Series Generation

[SDForger](https://arxiv.org/pdf/2505.17103) is a versatile methodology designed to enable the generation of time series using LLMs. Starting from a few observations of multiple time-series channels, the approach employs an efficient embedding representation to transform the time-series into tabular data, which is then converted into text. SDForger leverages fine-tuning to learn meaningful patterns within the computed embeddings. At inference time, it produces new textual embeddings decoded back into fully synthetic time series data that mimic the original data’s statistical properties and temporal dynamics.

## Structure of SDForger

This data builder supports generation defining the following parameters:

1. **Time-Series Pattern Extraction via Independant Component Analysis** \
   SDForger applies Fast Independant Component Anlysis (FICA) to extract dominant patterns in time series data and embed them into a structured tabular format.

2. **Template-Guided Textual Representation for LLM Fine-Tuning** \
   Utilizes a structured template to transform embedding tables into textual descriptions, preparing them for large language model (LLM) fine-tuning.

3. **Inference Step for Generation** \
   Employs a guided inference approach to generate structural embeddings.

4. **Refinement through Decoding and Filtering** \
   Implements a decoding mechanism followed by a filtering step to ensure high-quality output.

## Setup

This databuilder requires additional dependencies. To install, please run:

```shell
pip install -e ".[vllm]"
pip install -e ".[time_series]"
```

## Task specification

Default configuration for task is [here](../../../../tasks/public/time_series/bikesharing_univariate/task.yaml).

### Data specification

```yaml
seed_datastore:
  type: default
  data_path: ${DGT_DATA_DIR}/public/time_series/bikesharing_full.parquet

data_params:
  train_length: 5000 # length of the original time series that we want to use for augmentation
  train_samples: 1 # number of sample in the original data, 1 for univariate case
  augmentation_strategy: univariate # "multivariate", "multisample"
  train_channels:
    - cnt

# sdforger args
sdforger_params:
  k_bit: null
  min_outputs_to_generate: 50
  max_outputs_to_generate: 100
  inference_batch: 64 # number of input generated per inference
  norms_diversity_threshold: 1 # diversity score used to stop generation, set to 0 to disable
  embedding_type: "fica" # "fpc"
  embedding_dim: auto # int=3 used in paper, set 'auto' to dynamically determine the embedding diemnsion based on the variance_explained
  variance_explained: 0.7
  min_windows_number: 30 # minimum number of samples for training
  min_windows_length: 300 # length of the samples generated
  input_tokens_precision: 4 # decimal precision of input token values.
```

## Generators

Default configuration for generator used by the data builder is available [here](time_series.yaml).

```yaml
blocks:
  - name: trainer
    type: public/trainers/sdforger-tuning

    # General training args
    model_id_or_path: gpt2 #ibm-granite/granite-3.0-2b-base
    learning_rate: 0.00008
    num_train_epochs: 100
    per_device_train_batch_size: 32 # update this value as per your system memory
    seed: 42

# Target inference model will be loaded from the fine-tuned checkpoint above
target:
  type: vllm # Use only vllm as the lm provider type
  dtype: float32
  trust_remote_code: true
  ignore_mismatched_sizes: true
  temperature: 1.3
  max_tokens: 1000
```

## Usage

To try out the databuilder, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/time_series/bikesharing_univariate/task.yaml # univariate
python -m fms_dgt.public --task-paths ./tasks/public/time_series/bikesharing_multivariate/task.yaml # multivariate
python -m fms_dgt.public --task-paths ./tasks/public/time_series/nn5_multisample/task.yaml # multisample
```

## Output: Generated Time Series

If successful, a plot of generated time-series will be saved in `output/<task_name>/plot_generated_data.pdf`

## Contributors

**Authors and Maintainers**: Cécile Rousseau, Tobia Boschi, Dhaval Salwala
