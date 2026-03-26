# DiGiT Studio

DiGiT Studio is a local run-monitoring UI that lets you inspect active and completed runs without leaving your machine. It requires only Python: no Node.js runtime is needed after installation.

## Starting Studio

```bash
digit-studio
```

Studio reads `DGT_OUTPUT_DIR` from your `.env` file. You can override it for a specific session with `--output-dir`:

```bash
digit-studio --output-dir ./some-other-output
```

Once running, open [http://localhost:4720](http://localhost:4720) in your browser.

To stop a running instance:

```bash
digit-studio stop
```

## What Studio shows

Studio polls your output directory and telemetry files and displays the following for each run.

### Run list

The sidebar lists every run found under the output directory, with its status (`running`, `completed`, or `errored`), start time, and duration.

### Task card

The configuration used for the run: task name, databuilder, model, blocks, and postprocessors.

### Result

The final outcome record from `task_results.jsonl`, including status, timing, and any metrics emitted by the databuilder.

### Data points

The generated data files for the run, split across four tabs:

| Tab | File |
| --- | ---- |
| Intermediate | `data.jsonl` |
| Post-processed | `postproc_data_<n>.jsonl` |
| Final | `final_data.jsonl` |
| Formatted | `formatted_data.jsonl` |

### Log

The full run log, read from `logs.jsonl` (or `.log` files for older runs).

### Stats

Generation and postprocessing metrics derived from telemetry, when `DGT_TELEMETRY_DIR` is set. Includes per-epoch acceptance rates, throughput time series, and token usage with optional cost estimates.

## CLI reference

```bash
digit-studio --help
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--output-dir` | `$DGT_OUTPUT_DIR` | Output directory to monitor |
| `--telemetry-dir` | `$DGT_TELEMETRY_DIR` | Telemetry directory for stats and token usage |
| `--port` | `4720` | Port to listen on |

## Environment variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `DGT_OUTPUT_DIR` | `output/` | Output directory to monitor |
| `DGT_TELEMETRY_DIR` | `telemetry/` | Telemetry directory for generation stats and token usage |
| `DIGIT_STUDIO_PORT` | `4720` | Port Studio listens on |

## Cancelling a run

Select a running run in the sidebar and click **Stop** in the detail view. Studio sends a `SIGINT` to the run process, which triggers a clean shutdown.

## Development setup

### Prerequisites

Install Node.js via a version manager. [nvm](https://github.com/nvm-sh/nvm) is recommended:

```bash
nvm install
nvm use
```

The `.nvmrc` in the `studio/` directory pins the required Node version. Then install dependencies:

```bash
cd studio && npm install
```

### Running the dev server

```bash
cd studio
npm run dev
```

This starts the FastAPI backend on port 4720 (with auto-reload on Python file changes) and the Next.js dev server on port 3000. Open [http://localhost:3000](http://localhost:3000). All `/api/*` requests are proxied automatically to the backend.

A `.env` file at the repo root with `DGT_OUTPUT_DIR` and `DGT_TELEMETRY_DIR` set is required for the backend to find your runs.

To build and commit the static frontend after making UI changes:

```bash
cd studio
npm run build
```

Commit the updated `studio/dist/` directory alongside your code changes.
