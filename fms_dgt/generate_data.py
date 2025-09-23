# Standard
from datetime import datetime
from typing import Dict, List, Optional
import gc
import logging
import os

# Third Party
from dotenv import load_dotenv

# Local
from fms_dgt import SRC_DGT_DIR
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import add_namespace_to_searchable_dirs, get_data_builder
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.constants import (
    BLOCKS_KEY,
    DGT_ENV_VARS,
    RAY_CONFIG_KEY,
    RUNNER_CONFIG_KEY,
    TASK_NAME_KEY,
)
from fms_dgt.index import DataBuilderIndex
from fms_dgt.utils import dgt_logger
import fms_dgt.utils as utils


def generate_data(
    task_kwargs: Dict,
    builder_kwargs: Dict,
    task_paths: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    include_builder_paths: Optional[List[str]] = None,
    build_id: Optional[str] = None,
    ray_address: Optional[str] = None,
    debug: Optional[bool] = False,
    namespaces: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
):
    """Generate data for a set of tasks using their respective data builders

    Args:
        task_kwargs (Dict): A dictionary of keyword arguments to pass to each task.
        builder_kwargs (Dict): A dictionary of keyword arguments to pass to each data builder.
        task_paths (Optional[List[str]], optional): A list of paths to data files.
        config_path (Optional[str], optional): A path to a configuration file.
        include_builder_paths (Optional[List[str]], optional): A list of paths to search for data builders.
        build_id (Optional[str], optional): An ID to associate with all of the tasks executed in this run.
        ray_address (Optional[str], optional): An address to connect ray.init to.
        debug (Optional[bool], optional): Boolean flag to enable debugging. Defaults to False.
        namespaces (Optional[List[str]], optional): List of namespaces to initialize registry from.
        env_vars (Optional[Dict[str, str]], optional): Environment variables to add.
    """
    # Step 1: Update logging level, if requested
    if debug:
        dgt_logger.setLevel(logging.DEBUG)

    # Step 2: Initialize default builder paths
    if include_builder_paths is None:
        include_builder_paths = []

    # Step 3: Configure databuilder paths based on namespace
    for namespace in ["core"] + (namespaces or []):
        # Step 3.a: Add namespace to registry
        add_namespace_to_searchable_dirs(namespace)

        # Step 3.b: Get databuilder directory
        databuilders_dir = os.path.join(SRC_DGT_DIR, namespace, "databuilders")

        # Step 3.c: Add databuilders directory into databuilder paths, if applicable
        if databuilders_dir not in include_builder_paths:
            include_builder_paths.append(databuilders_dir)

    # Step 4: Configure environment variables
    # Step 4.a: Load environment variables from ".env"
    load_dotenv()

    # Step 4.b: Set environment variables, if not set already
    for k, v in {**DGT_ENV_VARS, **(env_vars or dict())}.items():
        if k not in os.environ:
            os.environ[k] = v

    # Step 5: Prepare task paths
    # Step 5.a: Initialize necessary variables
    task_paths = task_paths or []
    builder_overrides = None
    task_overrides = dict()

    # Step 5.b: If config path is provided
    if config_path:
        # Step 5.b.i: Load config
        (
            addlt_task_paths,
            builder_overrides,
            task_overrides,
        ) = utils.load_joint_config(config_path)

        # Step 5.b.ii: Extend task paths
        task_paths.extend(addlt_task_paths)

    # Step 5.c: Report missing task or config paths
    if not task_paths and not config_path:
        raise ValueError(
            "One of ['data-paths', 'task-paths', 'config-path'] must be provided in the arguments"
        )

    # Step 5.c: Retain unique task paths
    task_paths = list(set(task_paths))

    # Step 6: Load task configurations
    task_inits = []
    for task_path in task_paths:
        # Step 6.a.i: Verify task path exists
        if task_path and os.path.exists(task_path):
            for task_init in utils.read_tasks(task_path):
                task_init = {
                    **task_init,
                    **task_overrides.get(task_init["task_name"], dict()),
                }
                task_inits.append(task_init)
        else:
            raise FileExistsError(f"Error: task path ({task_path}) does not exist.")

    # capture tasks specified only in config overrides
    for task_name, task_override in task_overrides.items():
        if not any(
            [task_init.get(TASK_NAME_KEY) == task_name for task_init in task_inits]
        ):
            dgt_logger.info(
                "Task '%s' in config was not found in task files, it will be created from the config",
                task_name,
            )
            # explicitly add name field here
            task_inits.append({**task_override, TASK_NAME_KEY: task_name})

    # Step 7: Collate databuilders from task configurations
    # Step 7.a: Form requested databuilders list
    requested_databuilder_names = [t["data_builder"] for t in task_inits]

    # Step 7.b: Initialize databuilder index
    databuilder_index = DataBuilderIndex(
        include_builder_paths=include_builder_paths,
    )

    # Step 7.c: Identify available databuilders
    available_databuilder_names = databuilder_index.match_builders(
        requested_databuilder_names
    )
    dgt_logger.debug("Available databuilders: %s", available_databuilder_names)

    # Step 7.d: Identify missing databuilders
    missing_databuilder_names = set(
        [
            databuilder
            for databuilder in requested_databuilder_names
            if databuilder not in available_databuilder_names and "*" not in databuilder
        ]
    )

    # Step 7.e: Report missing databuilders, if applicable
    if missing_databuilder_names:
        raise ValueError(
            f"Builder specifications not found: [{', '.join(missing_databuilder_names)}]"
        )

    # Step 7.f: Load databuilder configurations
    available_databuilder_configurations = list(
        databuilder_index.load_builder_configs(
            available_databuilder_names, config_overrides=builder_overrides
        ).items()
    )

    # Step 8: Execute tasks for each databuilder
    for builder_name, builder_cfg in available_databuilder_configurations:
        # Step 8.a: Get databuilder information
        builder_info = databuilder_index.builder_index[builder_name]
        builder_dir = builder_info.get("builder_dir")
        if isinstance(builder_cfg, tuple):
            _, builder_cfg = builder_cfg
            if builder_cfg is None:
                continue

        dgt_logger.debug("Builder config for %s: %s", builder_name, builder_cfg)

        # Step 8.b: Prepare databuilder arguments
        all_builder_kwargs = {
            "config": builder_cfg,
            # Step 8.b.i: Prepare task arguments for tasks associated to the current databuilder
            "task_kwargs": [
                {
                    # Step 8.b.i.*: Prepare task card
                    "task_card": TaskRunCard(
                        task_name=task_init.get("task_name"),
                        databuilder_name=task_init.get("data_builder"),
                        task_spec={"task_init": task_init, "task_kwargs": task_kwargs},
                        databuilder_spec=utils.load_nested_paths(
                            builder_cfg, builder_dir
                        ),
                        build_id=build_id,
                        save_formatted_output=task_kwargs.get("save_formatted_output"),
                    ),
                    # Step 8.b.i.**: Prepare task runner configurations
                    RUNNER_CONFIG_KEY: {
                        **task_init.get(RUNNER_CONFIG_KEY, dict()),
                        **task_kwargs,
                    },
                    **{
                        k: v
                        for k, v in task_init.items()
                        if k not in [RUNNER_CONFIG_KEY]
                    },
                }
                for task_init in task_inits
                if task_init["data_builder"] == builder_name
            ],
            **builder_kwargs,
        }

        # Step 8.c: Ray initialization, if necessary
        ray_initialized = _init_ray(builder_cfg, ray_address)

        # Step 8.d: Initialize databuilder
        data_builder: DataBuilder = None

        # Step 8.d.i: Look into already initialized databuilders
        try:
            data_builder = get_data_builder(builder_name, **all_builder_kwargs)
        except KeyError as e:
            if f"Attempted to load data builder '{builder_name}'" not in str(e):
                raise e

        # Step 8.d.ii: initialize, if necessary
        if data_builder is None:
            utils.import_builder(builder_dir)
            data_builder = get_data_builder(builder_name, **all_builder_kwargs)

        # Step 8.e: Trigger tasks execution for the current databuilder
        data_builder.record_run_results(
            update={
                "PID": os.getpid(),
                "status": "running",
                "start_time": int(datetime.now().timestamp()),
                "end_time": None,
            }
        )
        try:
            data_builder.execute_tasks()
        # pylint: disable=broad-exception-caught
        except Exception as e:
            data_builder.record_run_results(
                update={
                    "status": "errored",
                    "end_time": int(datetime.now().timestamp()),
                    "message": str(e),
                }
            )

            # Raise exception
            raise e

        # Step 8.f: Cleanup databuilder
        data_builder.close()
        del data_builder

        # Step 8.g: Cleanup ray
        if ray_initialized:
            ray.shutdown()  # type: ignore

        # Step 8.h: Pre-cautionary garbage collection
        gc.collect()


def _init_ray(builder_kwargs, ray_address) -> bool:
    for block_cfg in builder_kwargs.get(BLOCKS_KEY):
        if block_cfg.get(RAY_CONFIG_KEY):
            # Third Party
            import ray

            # pylint: disable=pointless-string-statement
            """
            ray.init(
                address: str | None = None, *, num_cpus: int | None = None,
                num_gpus: int | None = None, resources: Dict[str, float] | None = None,
                labels: Dict[str, str] | None = None, object_store_memory: int | None = None,
                local_mode: bool = False, ignore_reinit_error: bool = False, include_dashboard: bool | None = None,
                dashboard_host: str = '127.0.0.1', dashboard_port: int | None = None,
                job_config: ray.job_config.JobConfig = None, configure_logging: bool = True,
                logging_level: int = 'info', logging_format: str | None = None,
                logging_config: LoggingConfig | None = None, log_to_driver: bool | None = None,
                namespace: str | None = None,
                runtime_env: Dict[str, Any] | RuntimeEnv | None = None, storage: str | None = None,
                **kwargs)
            """
            ray.init(address=ray_address)
            return True
    return False
