# Configuration

DiGiT runtime allows setting certain common options via command line arguments

## `--num-outputs-to-generate` <span style="color:#24a148; font-size:16px">[Default = 2]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --num-outputs-to-generate 10
```

You can easily control number of examples generated during a single run. We recommend starting with a small number (50, 100 or 500) during development and testing phase.


## `--config-path` <span style="color:#24a148; font-size:16px">[Default = None]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --config-path <path-to-config, e.g., ./configs/rits_fc_db>
```

Often times you will find yourself wanting to override the databuilder config without directly modifying it. You can specify an override config using the above flag.

## `--include-namespaces` <span style="color:#24a148; font-size:16px">[Default = None]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --include-namespaces granite
```

By default, the only namespaces loaded will be those specified in `fms_dgt.<namespace>.__main__`. You can specify any additional namespaces to consider (e.g., if a `block` from another namespace needs to be used but is not directly imported by the code) with the above flag

## `--output-dir` <span style="color:#24a148; font-size:16px">[Default = output/{task_name}]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --output-dir <PATH>
```

## `--restart-generation` <span style="color:#24a148; font-size:16px">[Default = False]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --restart-generation
```

DiGiT runtime tries to continue generation as long as the number of generated examples in `output-dir` is smaller than the number specified via `num-outputs-to-generate`. To delete previously generated example use `restart-generation` argument. 
> **WARNING**
> This is a destructive action and cannot be reversed.

## `--max-gen-requests` <span style="color:#24a148; font-size:16px">[Default = 100000]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --max-gen-requests 50
```

DiGiT runtime iterates over combination of seed examples and generated synthetic examples till it meets requirements set by `num-outputs-to-generate`. This could potentially lead to an infinite loop due to factors like faulty databuilder design, lack of diversity of seed examples and limited capabilities of a teacher model. `max-gen-requests` provides an escape hatch in such case where DiGiT runtime breaks iteration loop once the value set by `max-gen-requests` is reached.

## `--max-stalled-requests` <span style="color:#24a148; font-size:16px">[Default = 5]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --max-stalled-requests 10
```
DiGiT runtime iterates over combination of seed examples and generated synthetic examples till it meets requirements set by `num-outputs-to-generate`. This could potentially lead to an infinite loop due to factors like large language model (LLM) engine failures. `max-stalled-requests` provides an escape hatch in such case where DiGiT runtime breaks iteration loop once the value set by `max-stalled-requests` is reached.

## `--seed-batch-size` <span style="color:#24a148; font-size:16px">[Default = 100]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --seed-batch-size 50
```
DiGiT runtime creates a diverse batch of in-context learning examples (demonstrations) via combining seed examples and generated synthetic examples. `seed-batch-size` specifies the maximum number of seed examples to sequentially sample for a given batch.

## `--machine-batch-size` <span style="color:#24a148; font-size:16px">[Default = 10]</span>

```sh
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --machine-batch-size 20
```
DiGiT runtime creates a diverse batch of in-context learning examples (demonstrations) via combining seed examples and generated synthetic examples. `machine-batch-size` specifies the number of generated synthetic examples to randomly sample for a given batch.
