# Time Series Generation



SDForger is a versatile methodology designed to enable the generation of time series using LLMs. Starting from a few observations of multiple time series, the approach employs an efficient embedding representation to transform the time series into tabular data, which is then converted into text. SDForger leverages fine-tuning to learn meaningful patterns within the computed embeddings. At inference time, it produces new textual embeddings decoded back into fully synthetic time series data that mimic the original data’s statistical properties and temporal dynamics.



## Structure of SDForger



This data builder supports generation defining the following parameters:



1. **Time-Series Pattern Extraction via Functional PCA** \

   SDForger applies functional principal components analysis to extract dominant patterns in time series data and embed them into a structured tabular format.



2. **Template-Guided Textual Representation for LLM Fine-Tuning** \

   Utilizes a structured template to transform embedding tables into textual descriptions, preparing them for large language model (LLM) fine-tuning.



3. **Inference Step for Generation** \

   Employs a guided inference approach to generate structural embeddings.



4. **Refinement through Decoding and Filtering** \

   Implements a decoding mechanism followed by a filtering step to ensure high-quality output.



## Setup



This databuilder requires additional dependencies. To install, please run:



```shell

pip install ".[time_series]"

```



## Data specification



Default configuration for dataloader is [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/time_series/task.yaml).



```yaml

runner_config:

  num_outputs_to_generate: 200

  seed_batch_size: 15800

```



- `seed_batch_size`: end point of our original time series that we want to use for augmentation, \

  !!! be careful, we require: `train_length` < `seed_batch_size` < length of original time series !!!



```yaml

seed_datastore:

  type: default

  data_path: ${DGT_DATA_DIR}/time_series/datasource/univariate/bikesharing_full.parquet

  data_params:

    standardize: true

    train_length: 5000

    train_samples: 1

    train_channels:

      - cnt

```



- `data_path`: our example uses the bikesharing dataset

- `standardize`: we recommand and implement a standardscaling normalization technique

- `train_length`: length of the original time series that we want to use for augmentation

- `train_samples`: number of instances, 1 for most cases

- `train_channels`: columns of the dataset that we want to augment



Sample data records in the parquet file



```json

{'Unnamed: 0': 0, 'instant': 1, 'dteday': '2011-01-01 00:00:00', 'season': 1, 'yr': 0, 'mnth': 1, 'hr': 0, 'holiday': 0, 'weekday': 6, 'workingday': 0, 'weathersit': 1, 'temp': 0.24, 'atemp': 0.2879, 'hum': 0.81, 'windspeed': 0.0, 'casual': 3, 'registered': 13, 'cnt': 16}

{'Unnamed: 0': 1, 'instant': 2, 'dteday': '2011-01-01 01:00:00', 'season': 1, 'yr': 0, 'mnth': 1, 'hr': 1, 'holiday': 0, 'weekday': 6, 'workingday': 0, 'weathersit': 1, 'temp': 0.22, 'atemp': 0.2727, 'hum': 0.8, 'windspeed': 0.0, 'casual': 8, 'registered': 32, 'cnt': 40}

```



## Generators



Default configuration for generator used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/time_series/time_series.yaml).



```yaml

target:

  name: time_series_inference_model

  blocks:

    - name: llm1

      type: vllm # Use only vllm as the inference model is first fine-tuned and then loaded from the specified location.

      dtype: float32

      trust_remote_code: true,

      ignore_mismatched_sizes: true,

      temperature: 1.3

      max_length: 1000

      # For RITS, vLLM, OpenAI "/chat/completions"

      max_completion_tokens: 1024

      # For RITS, vLLM, OpenAI "/completions"

      max_tokens: 1024

      model_id_or_path: gpt2 # ibm-granite/granite-3.1-2b-instruct

```



- `model_id_or_path`: LLM used at the core of SDForger for data augmentation (`ibm-granite/granite-3.1-2b-base` and `gpt2` are implemented)

- `temperature`: LLM temperature

- `max_length`: LLM max_length for generation



```yaml

- name: trainer1

    type: sdforger-tuning



    # hf training args

    seed: 42

    torch_dtype: float32

    learning_rate: 0.00008

    num_train_epochs: 100

    per_device_train_batch_size: 32 # update this value as per your system memory.



    # sdforger args

    sdforger_params:

      # training params

      k_bit: null

      norms_diversity_threshold: 0.99

      augmentation_strategy: univariate

      embedding_type: "fpc"

      embedding_dim: auto

      embedding_feature_type: continuous # continuous, categorical

      variance_explained: 0.5

      train_splitting: "minimize-overlap"

      min_windows_number: 30

      min_windows_length: 1120

      permute_input_tokens: true

      nums_input_tokens: 100

      init_input_tokens: false

      input_tokens_precision: 4

```



- `norms_diversity_threshold`: diversity score used to stop generation

- `augmentation_strategy`: "univariate" / "multivariate" / "multisample" (this is a univariate example of the augmentation of the channel cnt of the bikesharing dataset)

- `embedding_type`: "fpc", "fpc-filled"

- `embedding_dim`: auto

- `variance_explained`: target variance used to select automatically the embedding dimension

- `permute`: True (permutation of embedding in finetuning data to remove positional information)

- `init_value`: False (no iniciation of value in inference text prompt )

- `train_splitting`: "minimize-overlap" for strategy of minimizing

- `min_windows_number`: 30

- `min_windows_length`: 1120

- `permute_input_tokens`: whether to permute input tokens

- `nums_input_tokens`: number of input tokens per generation

- `init_input_tokens`: whether to add initial values in input tokens

- `input_tokens_precision`: decimal precision of input token values



## Output: Generated Time Series



Generated Time Series are saved in ./fms-dgt-internal/output/time-series-generation/time_series.csv



## Contributors



**Authors and Maintainers**: Cécile Rousseau, Dhaval Salwala, Tobia Boschi
