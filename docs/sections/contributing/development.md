# Development

## Set up your dev environment

The following tools are required:

- [git](https://git-scm.com)
- [python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

You can setup your dev environment using [tox](https://tox.wiki/en/latest/), an environment orchestrator which allows for setting up environments for and invoking builds, unit tests, formatting, linting, etc. Install tox with:

```sh
pip install tox
```

If you want to manage your own virtual environment instead of using `tox`, you can install `FMS SDG` and all dependencies with:

```sh
pip install .
```

## Testing

Before pushing changes to GitHub, you need to run the tests as shown below. They can be run individually as shown in each sub-section or can be run with the one command:

```shell
tox
```

### Unit tests

Unit tests are enforced by the CI system. When making changes, run the tests before pushing the changes to avoid CI issues.

Running unit tests can be done with:

```sh
tox -e unit
```

By default, all tests found within the `tests` directory are run. However, specific unit tests can run by passing filenames, classes and/or methods to `pytest` using tox positional arguments. The following example invokes a single test method `test_generate_batch` within the `Test_GenAIGenerator` class that is declared in the `tests/generators/test_genai.py` file:

```shell
tox -e unit -- tests/generators/test_genai.py::Test_GenAIGenerator::test_generate_batch
```

### Coding style

FMS SDG follows the python [pep8](https://peps.python.org/pep-0008/) coding style. The coding style is enforced by the CI system, and your PR will fail until the style has been applied correctly.

We use [pre-commit](https://pre-commit.com/) to enforce coding style using [black](https://github.com/psf/black), [prettier](https://github.com/prettier/prettier) and [isort](https://pycqa.github.io/isort/).

You can invoke formatting with:

```sh
tox -e fmt
```

In addition, we use [Ruff](https://docs.astral.sh/ruff/) to perform static code analysis of the code.

You can invoke the linting with the following command

```sh
tox -e lint
```

## Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues with the [`good first issue` label](https://github.ibm.com/conversational-ai/instructlab-datagen-pipelines/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these should only require a few lines of code and are good targets if you're just starting contributing.
- Issues with the [`help wanted` label](https://github.ibm.com/conversational-ai/instructlab-datagen-pipelines/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these range from simple to more complex, but are generally things we want but can't get to in a short time frame.
