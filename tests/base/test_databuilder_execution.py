# Standard
import gc
import multiprocessing
import os
import shutil
import time

# Third Party
import pytest

# Local
from fms_dgt import INTERNAL_DGT_DIR
from fms_dgt.__main__ import generate_data, parse_cmd_line
from fms_dgt.base.registry import REGISTRATION_SEARCHABLE_DIRECTORIES

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

to_execute = [
    #
    # simple
    #
    (
        "core",
        "simple",
        f"--task-paths {os.path.join(INTERNAL_DGT_DIR, 'tasks/core/simple/logical_reasoning/causal/task.yaml')} --num-outputs-to-generate 1 --output-dir {_OUTPUT_DIR}",
        50,
    ),
]


@pytest.mark.parametrize("namespace,data_builder_name,cmd_line_args,timeout", to_execute)
def test_data_builders(namespace: str, data_builder_name: str, cmd_line_args: str, timeout: int):
    """This file contains execution tests for each data builder (in the same way it
    would be called from the command-line). To add a new test, add your data builder,
    its command-line arguments, and a timeout to the 'to_execute' list. The command line
    arguments should result in a reasonably quick execution and the timeout value should
    indicate the maximum allowable time it takes to run the command.

    NOTE: this assumes the default settings of your databuilder config are what you want to use for testing

    Args:
        data_builder_name (str): name of databuilder to be tested
        cmd_line_args (str): command-line argument string
        timeout (int): time in seconds to allocate to test
    """

    arg_list = cmd_line_args.split()
    base_args, builder_kwargs, task_kwargs = parse_cmd_line(arg_list)

    if os.path.exists(task_kwargs.get("output_dir")):
        shutil.rmtree(task_kwargs.get("output_dir"))

    p = multiprocessing.Process(
        target=execute_db_test,
        args=(namespace, base_args, builder_kwargs, task_kwargs),
    )

    p.start()

    # wait for 'timeout' seconds or until process finishes
    p.join(timeout)

    assert not p.is_alive(), f"'{data_builder_name}' data builder took to long to execute"

    assert p.exitcode == 0, f"'{data_builder_name}' data builder failed during execution"

    # if thread is still active
    if p.is_alive():
        p.terminate()
        time.sleep(1)
        if p.is_alive():
            p.kill()
            time.sleep(1)
        gc.collect()

    time.sleep(5)

    if os.path.exists(task_kwargs.get("output_dir")):
        shutil.rmtree(task_kwargs.get("output_dir"))


def execute_db_test(
    namespace: str,
    base_args: dict,
    builder_kwargs: dict,
    task_kwargs: dict,
):
    namespaces = [namespace] if namespace != "core" else []
    namespaces.insert(0, "core")
    namespaces.extend([x for x in base_args.pop("include_namespaces", []) if x not in namespaces])

    REGISTRATION_SEARCHABLE_DIRECTORIES.clear()
    generate_data(task_kwargs, builder_kwargs, namespaces=namespaces, **base_args)
