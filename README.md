# FMS-DGT

DGT (pronounced "digit") is a framework that enables different algorithms and models to be used to generate synthetic data.

![Python Version](https://badgen.net/static/Python/3.10.15-3.12/blue?icon=python)
[![Code style: black](https://badgen.net/static/Code%20Style/black/black)](https://github.com/psf/black)
![GitHub License](https://badgen.net/static/license/Apache%202.0/green)

\*\*[Setup](#setup) | [Usage](#usage)

This is the main repository for DiGiT, our **D**ata **G**eneration and **T**ransformation framework.

## Setup

First clone the repository

```bash
git clone git@github.com:IBM/fms_dgt.git
cd fms_dgt
```

Now set up your virtual environment. We recommend using a Python virtual environment with Python >=3.10.15 and <3.13.x. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To install packages, we recommend the following

```bash
pip install -e ".[all]"
```

> [!IMPORTANT]
> Please install the pre-commit hooks to adhere with code hygiene standards
>
> ```bash
> pip install pre-commit
> pre-commit install
> ```

For whichever of various API services you plan on using, you need to add configurations to `.env` file. Copy the `.env.example` as `.env` and add your KEYS as follows:

```yaml
# watsonx
WATSONX_API_KEY=<WatsonX key goes here>
WATSONX_PROJECT_ID=<Project env variable>

# openai
OPENAI_API_KEY=<OPENAI key goes here>
```

## Usage

To test whether you have been successful, run the following operation that references a databuilder.

- If you have set up a `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`

```bash
python -m fms_dgt.core --task-paths ./tasks/core/simple/logical_reasoning/causal --restart-generation
```

If successful, you should see the outputs of the command in the `./output` directory
