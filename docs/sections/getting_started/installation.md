# Installation

Before you start, please make sure you have **Python 3.10+** available.

Building DiGiT from source lets you make changes to the code base. To install from source, clone the repository and install with the following commands:

```bash
git clone git@github.ibm.com:DGT/fms-dgt.git
cd fms-dgt
```

Now set up your virtual environment. We recommend using a Python virtual environment with Python 3.10+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html)

```bash
python3.10 -m venv ssdg_venv
source ssdg_venv/bin/activate
```


To install packages, we recommend starting off with the following

```bash
pip install -e .
pip install -e ".[granite, research]"
```

If you plan on contributing, install the pre-commit hooks to keep code formatting clean

```bash
pip install pre-commit
pre-commit install
```

> **NOTE**
>
> If you have used [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or another tool for Python version management, it is best to use the virtual environment associated with that tool instead. Otherwise, you may encounter module-not-found errors with modules from packages you have installed (as they are linked to you Python version management tool and not `venv`).

### Large Language Models (LLMs) Dependencies

DiGiT uses Large Language Models (LLMs) to generate synthetic data. Following LLM inference services are supported:

- [RITS](https://rits.fmaas.res.ibm.com/) <span style="color:#24a148">`Recommended`</span>
- [WatsonX](https://www.ibm.com/watsonx) <span style="color:#24a148">`Recommended`</span>
- [OpenAI](https://github.com/openai/openai-python)
- [vLLM](https://github.com/vllm-project/vllm) <span style="color:#f1c21b">`Experimental`</span>

Most of the aforementioned LLM inference services use environment variables to specify configuration settings. You can either export those environment variables prior to every run or save them in `.env` file at base of `fms-dgt-internal` repository directory.

```yaml
# RITS (Recommended)
RITS_API_KEY=""

# WatsonX (Recommended)
WATSONX_API_KEY=""
WATSONX_PROJECT_ID=""

# OpenAI (Third-party)
OPENAI_API_KEY=""
```

> **IMPORTANT**
>
> `RITS` is only available to Research division employees. Additionally, you will need to use `TUNNELALL` VPN to access `RITS`. To configure,
> 1. Open Cisco Secure Client
> 2. Type `sasvpn-dc.us.ibm.com/macOS-TUNNELALL`
> 3. Click Connect


Some LLM inference services are not installed with minimal installation.

> **NOTE**
> 
> To install vLLM dependencies ([requires Linux OS and CUDA](https://docs.vllm.ai/en/latest/getting_started/installation.html#requirements))
> 
> ```shell
> pip install -e ".[vllm]"
> ```

### Data Storage Dependencies

When using the DMF integration, you need to:

- Add configuration to `.env` file as follows:

```yaml
LAKEHOUSE_TOKEN=<DMF Lakehouse token goes here>
LAKEHOUSE_ENVIRONMENT=<PROD or STAGING>
```

- Follow the instructions [here](https://github.ibm.com/IBM-Research-AI/lakehouse-eval-api#lakehouse-token) to generate your lakehouse token.

- Install DMF Lakehouse dependencies as follows:

```shell
pip install ".[dmf-lakehouse]"
```

##  🐛 Known Issues

### Missing Python.h

Some libraries (e.g., vllm) require python-dev to function (you'll likely see an error about a missing Python.h file). If your python doesn't have python-dev (e.g., the default CCC version of python), you may run into issues. To solve this, we recommend working with python installed by miniforge

```bash
cd <directory-where-you-want-to-install-miniforge>
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
