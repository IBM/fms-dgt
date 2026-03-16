# Time Series Data Augmentation

The `time_series` databuilder generates synthetic time series data using **SDForger**, a method that extracts structural patterns from a real time series via fast independent component analysis (FICA), represents those patterns as token sequences, and uses a fine-tuned LLM to generate new sequences that preserve the original dynamics.

The result is augmented time series data that can be used to improve forecasting models in data-scarce settings.

!!! note
    This databuilder requires additional dependencies and a fine-tuned model checkpoint. See the prerequisites below before running.

## Prerequisites

Install the required extras:

```bash
pip install -e ".[vllm]"
pip install -e ".[time_series]"
```

The databuilder uses a vLLM-served fine-tuned checkpoint for generation. Refer to the [time series README](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/time_series) for instructions on obtaining or training the checkpoint.

## Example datasets

Three example tasks ship with the repository, covering the main augmentation strategies.

| Task                                                          | Strategy     | Dataset                             |
| ------------------------------------------------------------- | ------------ | ----------------------------------- |
| `tasks/public/time_series/bikesharing_univariate/task.yaml`   | Univariate   | DC bike sharing (single channel)    |
| `tasks/public/time_series/bikesharing_multivariate/task.yaml` | Multivariate | DC bike sharing (multiple channels) |
| `tasks/public/time_series/nn5_multisample/task.yaml`          | Multisample  | NN5 competition dataset             |

## Univariate augmentation

Augment a single channel of the DC bike sharing dataset.

### Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/bikesharing_univariate/task.yaml \
  --restart
```

### Task specification

```yaml
task_name: time_series/bikesharing_univariate
task_description: Generate synthetic time series data.
created_by: IBM Research
data_builder: time_series

data:
  type: default
  data_path: ${DGT_DATA_DIR}/public/time_series/bikesharing_full.parquet

data_params:
  train_length: 5000 # number of time steps to use from the original series
  train_samples: 1 # 1 for univariate
  augmentation_strategy: univariate
  train_channels:
    - cnt # column name to augment

sdforger_params:
  embedding_type: fica
  embedding_dim: auto # auto selects dimension based on variance_explained
  variance_explained: 0.7
  min_windows_number: 30
  min_windows_length: 300
  min_outputs_to_generate: 50
  max_outputs_to_generate: 100
  inference_batch: 64
  norms_diversity_threshold: 1
  input_tokens_precision: 4
```

### How it works

1. **Pattern extraction:** FICA decomposes the input time series into independent structural components.
2. **Textual representation:** each component is encoded as a token sequence with controlled numerical precision.
3. **LLM inference:** the fine-tuned model generates new token sequences that follow the learned distribution.
4. **Refinement:** generated sequences are decoded, diversity-filtered, and written to the output store.

### Output

Generated data is written to `output/time_series/bikesharing_univariate/final_data.jsonl`. A plot of the generated series is also saved to `output/time_series/bikesharing_univariate/plot_generated_data.pdf` for visual inspection.

## Multivariate augmentation

To augment multiple correlated channels simultaneously, use the multivariate task:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/bikesharing_multivariate/task.yaml \
  --restart
```

Set `train_samples` to the number of channels and list each channel name under `train_channels` in the task YAML.

## Multisample augmentation

To augment a dataset containing multiple independent time series (for example, multiple store sales series), use the multisample task:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/nn5_multisample/task.yaml \
  --restart
```

## Next steps

- Read the [time series databuilder README](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/time_series) for full parameter documentation.
- To bring your own dataset, create a new task YAML pointing to your parquet file and adjust `data_params` to match your series length and channel structure.
