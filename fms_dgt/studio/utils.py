# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""File-reading and data-processing utilities for DiGiT Studio.

All functions are pure (no FastAPI imports, no global state).  The API layer
in server.py calls these and handles HTTP concerns.
"""

# Standard
from datetime import datetime, timedelta, timezone
import json
import math
import os
import signal
import time


# ===================================================================================
#                               JSONL PARSING
# ===================================================================================
def parse_jsonl(content: str) -> list:
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return parse_jsonl(f.read())


# ===================================================================================
#                               TIME / DURATION HELPERS
# ===================================================================================
def cast_duration_to_string(duration_secs: float) -> tuple[int, int, int, int]:
    """Return (days, hours, minutes, seconds) from a duration in seconds."""
    days = math.floor(duration_secs / 86400)
    duration_secs -= days * 86400
    hours = math.floor(duration_secs / 3600)
    duration_secs -= hours * 3600
    minutes = math.floor(duration_secs / 60)
    duration_secs -= minutes * 60
    seconds = math.floor(duration_secs)
    return days, hours, minutes, seconds


def format_duration(secs: float) -> str:
    days, hours, mins, seconds = cast_duration_to_string(secs)
    parts = []
    if days:
        parts.append(f"{days} days")
    if hours:
        parts.append(f"{hours} hours")
    if mins:
        parts.append(f"{mins} mins")
    parts.append(f"{seconds} sec")
    return " ".join(parts)


# ===================================================================================
#                               BUCKET / TIME-SERIES HELPERS
# ===================================================================================
def adaptive_bucket_secs(duration_ms: float) -> int:
    mins = duration_ms / 60000
    if mins < 1:
        return 5
    if mins < 10:
        return 15
    if mins < 60:
        return 60
    return 300


def to_bucket(iso_string: str, bucket_secs: int) -> str:
    """Snap an ISO timestamp to a grid of `bucket_secs` seconds."""
    base = iso_string[:19]
    d = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
    snapped = math.floor(d.timestamp() / bucket_secs) * bucket_secs
    return datetime.fromtimestamp(snapped, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def shift_bucket(iso_string: str, offset_secs: int) -> str:
    """Shift an ISO datetime string by offset_secs seconds."""
    base = iso_string[:19]
    d = datetime.fromisoformat(base).replace(tzinfo=timezone.utc) + timedelta(seconds=offset_secs)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


# ===================================================================================
#                               RUN DISCOVERY
# ===================================================================================
def walk(directory: str, name: str, root: str) -> list[dict]:
    """Recursively find run directories (those with task_card.jsonl + task_results.jsonl)."""
    runs = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return runs

    for filename in entries:
        file_path = os.path.join(directory, filename)
        if not os.path.isdir(file_path):
            continue

        child_files = os.listdir(file_path)
        run_name = f"{name}/{filename}" if name else filename

        if "task_card.jsonl" in child_files and "task_results.jsonl" in child_files:
            try:
                start_time_ts = os.path.getmtime(os.path.join(file_path, "task_card.jsonl"))
            except OSError:
                start_time_ts = time.time()

            start_time = start_time_ts
            duration = "-"
            status = "running"

            try:
                results_data = read_jsonl(os.path.join(file_path, "task_results.jsonl"))
                last = results_data[-1]
                start_time = last["start_time"]
                end_time = last.get("end_time") or time.time()
                duration = format_duration(end_time - start_time)
                status = last["status"]
            except Exception:
                try:
                    mtime = os.path.getmtime(os.path.join(file_path, "task_results.jsonl"))
                    duration = format_duration(mtime - start_time_ts)
                except Exception:
                    pass

            runs.append(
                {
                    "name": run_name,
                    "path": f"{root}/{run_name}",
                    "status": status,
                    "startTime": start_time,
                    "duration": duration,
                }
            )
        else:
            runs.extend(walk(file_path, run_name, root))

    return runs


# ===================================================================================
#                               RUN DETAIL LOADERS
# ===================================================================================
def load_task_cards(path: str) -> list[dict]:
    return read_jsonl(os.path.join(path, "task_card.jsonl"))


def load_log(path: str) -> str | None:
    # Prefer logs.jsonl (public-repo LogDatastoreHandler format).
    try:
        entries = []
        with open(os.path.join(path, "logs.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                entries.append(
                    f"{e['timestamp']} {e['level'].ljust(8)} [{e['logger']}:{e['lineno']}] {e['message']}"
                )
        if entries:
            return "\n".join(entries)
    except Exception:
        pass

    # Fall back to plain-text .log files (older internal-repo format, sorted by mtime).
    try:
        logs_dir = os.path.join(path, "logs")
        log_files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)),
        )
        if log_files:
            parts = []
            for f in log_files:
                with open(os.path.join(logs_dir, f), encoding="utf-8") as fh:
                    parts.append(fh.read())
            return "\n".join(parts)
    except Exception:
        pass

    return None


def load_log_since(path: str, offset: int) -> tuple[str, int]:
    """Return only log content written after ``offset`` bytes, plus the new offset.

    Works for logs.jsonl (structured) and plain-text .log fallback.  The offset
    is a byte position in the underlying source file so seeks are O(1).  Returns
    ("", offset) when nothing new is available.
    """
    # Try logs.jsonl first.
    logs_jsonl = os.path.join(path, "logs.jsonl")
    try:
        with open(logs_jsonl, "rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            if offset >= end:
                return "", offset
            fh.seek(offset)
            new_bytes = fh.read()
        lines = []
        for raw in new_bytes.decode("utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
                lines.append(
                    f"{e['timestamp']} {e['level'].ljust(8)} [{e['logger']}:{e['lineno']}] {e['message']}"
                )
            except Exception:
                lines.append(raw)
        return "\n".join(lines), end
    except FileNotFoundError:
        pass
    except Exception:
        return "", offset

    # Fall back to plain-text .log files — concatenate all, seek within combined content.
    try:
        logs_dir = os.path.join(path, "logs")
        log_files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)),
        )
        if not log_files:
            return "", offset
        # Build combined bytes lazily to find the right position.
        combined = b""
        for f in log_files:
            with open(os.path.join(logs_dir, f), "rb") as fh:
                combined += fh.read()
        new_bytes = combined[offset:]
        text = new_bytes.decode("utf-8", errors="replace")
        return text, len(combined)
    except Exception:
        return "", offset


def load_results(path: str) -> dict[str, dict]:
    return {str(r["PID"]): r for r in read_jsonl(os.path.join(path, "task_results.jsonl"))}


def load_data_points(path: str) -> dict:
    data: dict = {"intermediate": [], "postprocessed": {}, "final": [], "formatted": []}
    try:
        for filename in os.listdir(path):
            try:
                fp = os.path.join(path, filename)
                if filename == "data.jsonl":
                    data["intermediate"] = read_jsonl(fp)
                elif filename.startswith("postproc_data_") and filename.endswith(".jsonl"):
                    data["postprocessed"][filename[14:-6]] = read_jsonl(fp)
                elif filename == "final_data.jsonl":
                    data["final"] = read_jsonl(fp)
                elif filename == "formatted_data.jsonl":
                    data["formatted"] = read_jsonl(fp)
            except Exception:
                pass
    except Exception:
        pass
    return data


def _tail_jsonl(filepath: str, n: int) -> list:
    """Return the last ``n`` records from a JSONL file without reading the whole file."""
    try:
        with open(filepath, "rb") as fh:
            # Walk backwards collecting complete lines.
            fh.seek(0, 2)
            remaining = fh.tell()
            buf = b""
            lines: list[bytes] = []
            chunk = 4096
            while remaining > 0 and len(lines) <= n:
                read_size = min(chunk, remaining)
                remaining -= read_size
                fh.seek(remaining)
                buf = fh.read(read_size) + buf
                lines = buf.splitlines()
            # Keep only the last n non-empty lines.
            tail = [line for line in lines if line.strip()][-n:]
        return [json.loads(line) for line in tail]
    except Exception:
        return []


def _count_jsonl_lines(filepath: str) -> int:
    with open(filepath, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def load_data_points_tail(path: str, n: int = 25) -> dict:
    """Return the last ``n`` records from each data file, plus total counts."""
    data: dict = {
        "intermediate": [],
        "intermediateTotal": 0,
        "postprocessed": {},
        "postprocessedTotal": {},
        "final": [],
        "finalTotal": 0,
        "formatted": [],
        "formattedTotal": 0,
    }
    try:
        for filename in os.listdir(path):
            fp = os.path.join(path, filename)
            try:
                if filename == "data.jsonl":
                    data["intermediate"] = _tail_jsonl(fp, n)
                    data["intermediateTotal"] = _count_jsonl_lines(fp)
                elif filename.startswith("postproc_data_") and filename.endswith(".jsonl"):
                    key = filename[14:-6]
                    data["postprocessed"][key] = _tail_jsonl(fp, n)
                    data["postprocessedTotal"][key] = _count_jsonl_lines(fp)
                elif filename == "final_data.jsonl":
                    data["final"] = _tail_jsonl(fp, n)
                    data["finalTotal"] = _count_jsonl_lines(fp)
                elif filename == "formatted_data.jsonl":
                    data["formatted"] = _tail_jsonl(fp, n)
                    data["formattedTotal"] = _count_jsonl_lines(fp)
            except Exception:
                pass
    except Exception:
        pass
    return data


def load_data_points_page(path: str, page: int, page_size: int) -> dict:
    """Return a paginated slice from each data file, plus total counts."""
    data: dict = {
        "intermediate": [],
        "intermediateTotal": 0,
        "postprocessed": {},
        "postprocessedTotal": {},
        "final": [],
        "finalTotal": 0,
        "formatted": [],
        "formattedTotal": 0,
    }
    try:
        for filename in os.listdir(path):
            fp = os.path.join(path, filename)
            try:
                if filename == "data.jsonl":
                    all_records = read_jsonl(fp)
                    data["intermediateTotal"] = len(all_records)
                    start = page * page_size
                    data["intermediate"] = all_records[start : start + page_size]
                elif filename.startswith("postproc_data_") and filename.endswith(".jsonl"):
                    key = filename[14:-6]
                    all_records = read_jsonl(fp)
                    data["postprocessedTotal"][key] = len(all_records)
                    start = page * page_size
                    data["postprocessed"][key] = all_records[start : start + page_size]
                elif filename == "final_data.jsonl":
                    all_records = read_jsonl(fp)
                    data["finalTotal"] = len(all_records)
                    start = page * page_size
                    data["final"] = all_records[start : start + page_size]
                elif filename == "formatted_data.jsonl":
                    all_records = read_jsonl(fp)
                    data["formattedTotal"] = len(all_records)
                    start = page * page_size
                    data["formatted"] = all_records[start : start + page_size]
            except Exception:
                pass
    except Exception:
        pass
    return data


def resolve_pid(task_card: dict, results: dict[str, dict]) -> str:
    if task_card.get("process_id") is not None:
        return str(task_card["process_id"])
    sorted_results = sorted(results.values(), key=lambda r: r["start_time"], reverse=True)
    return str(sorted_results[0]["PID"])


def cancel_run(path: str) -> None:
    task_cards = load_task_cards(path)
    task_card = task_cards[-1]
    pid = None
    if task_card.get("process_id"):
        pid = int(task_card["process_id"])
    if pid is None:
        results = load_results(path)
        running = sorted(
            [r for r in results.values() if r.get("status") == "running"],
            key=lambda r: r["start_time"],
            reverse=True,
        )
        if running:
            pid = int(running[0]["PID"])
    if pid is not None:
        os.kill(pid, signal.SIGINT)


# ===================================================================================
#                               EPOCH TIMINGS (from traces.jsonl)
# ===================================================================================
def load_epoch_timings(run_id: str, telemetry_dir: str) -> dict[int, dict]:
    result: dict[int, dict] = {}
    traces_path = os.path.join(telemetry_dir, "traces.jsonl")
    if not os.path.exists(traces_path):
        return result
    try:
        spans = read_jsonl(traces_path)
    except Exception:
        return result

    for s in spans:
        if s.get("run_id") != run_id or s.get("epoch") is None:
            continue
        epoch = int(s["epoch"])
        ms = s.get("duration_ms") or 0
        existing = result.get(epoch, {"generationMs": 0, "postprocessingMs": 0})
        if s.get("span_name") == "dgt.epoch":
            result[epoch] = {**existing, "generationMs": ms}
        elif s.get("span_name") == "dgt.postprocessing":
            result[epoch] = {**existing, "postprocessingMs": ms}

    # Subtract postprocessing from epoch to get pure generation time.
    for epoch, timing in result.items():
        result[epoch] = {
            **timing,
            "generationMs": max(0, timing["generationMs"] - timing["postprocessingMs"]),
        }
    return result


# ===================================================================================
#                               GENERATION STATS (from events.jsonl)
# ===================================================================================
def load_generation_stats(run_id: str, telemetry_dir: str) -> dict | None:
    events_path = os.path.join(telemetry_dir, "events.jsonl")
    if not os.path.exists(events_path):
        return None
    try:
        events = read_jsonl(events_path)
    except Exception:
        return None

    run_events = [e for e in events if e.get("run_id") == run_id]
    epoch_map: dict[int, dict] = {}

    for e in run_events:
        if (
            e.get("event") == "postprocessing_finished"
            and e.get("epoch") is not None
            and e.get("task_counts")
        ):
            generated = sum(c.get("before") or 0 for c in e["task_counts"].values())
            survived = sum(c.get("after") or 0 for c in e["task_counts"].values())
            existing = epoch_map.get(
                e["epoch"], {"generated": 0, "survived": 0, "generationAttempts": 0}
            )
            epoch_map[e["epoch"]] = {**existing, "generated": generated, "survived": survived}

        if e.get("event") == "epoch_finished" and e.get("epoch") is not None:
            existing = epoch_map.get(
                e["epoch"], {"generated": 0, "survived": 0, "generationAttempts": 0}
            )
            epoch_map[e["epoch"]] = {
                **existing,
                "generationAttempts": e.get("generation_attempts") or 0,
            }

    if not epoch_map:
        return None

    epoch_timings = load_epoch_timings(run_id, telemetry_dir)
    series = [
        {
            "epoch": epoch,
            **data,
            "generationMs": epoch_timings.get(epoch, {}).get("generationMs", 0),
            "postprocessingMs": epoch_timings.get(epoch, {}).get("postprocessingMs", 0),
        }
        for epoch, data in sorted(epoch_map.items())
    ]

    total_generated = sum(b["generated"] for b in series)
    total_survived = sum(b["survived"] for b in series)
    last = series[-1]

    started_event = next((e for e in run_events if e.get("event") == "run_started"), None)
    finished_event = next((e for e in run_events if e.get("event") == "run_finished"), None)
    run_started_at = (
        started_event["timestamp"][:19]
        if started_event and started_event.get("timestamp")
        else None
    )
    run_finished_at = (
        finished_event["timestamp"][:19]
        if finished_event and finished_event.get("timestamp")
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    )

    start_ms = (
        datetime.fromisoformat(run_started_at).replace(tzinfo=timezone.utc).timestamp() * 1000
        if run_started_at
        else None
    )
    end_ms = datetime.fromisoformat(run_finished_at).replace(tzinfo=timezone.utc).timestamp() * 1000
    bucket_secs = adaptive_bucket_secs(end_ms - start_ms if start_ms is not None else 0)

    zero_anchor = (
        shift_bucket(to_bucket(run_started_at, bucket_secs), -bucket_secs)
        if run_started_at
        else None
    )

    postproc_events = sorted(
        [
            e
            for e in run_events
            if e.get("event") == "postprocessing_finished"
            and e.get("timestamp")
            and e.get("task_counts")
        ],
        key=lambda e: e["timestamp"],
    )

    cum_generated = 0
    cum_survived = 0
    time_series = []
    if zero_anchor:
        time_series.append({"timestamp": zero_anchor, "cumGenerated": 0, "cumSurvived": 0})
    for e in postproc_events:
        cum_generated += sum(c.get("before") or 0 for c in e["task_counts"].values())
        cum_survived += sum(c.get("after") or 0 for c in e["task_counts"].values())
        time_series.append(
            {
                "timestamp": to_bucket(e["timestamp"], bucket_secs),
                "cumGenerated": cum_generated,
                "cumSurvived": cum_survived,
            }
        )

    return {
        "totalGenerated": total_generated,
        "totalSurvived": total_survived,
        "lastEpoch": last["epoch"],
        "lastEpochGenerated": last["generated"],
        "lastEpochSurvived": last["survived"],
        "lastEpochGenerationAttempts": last["generationAttempts"],
        "series": series,
        "timeSeries": time_series,
        "source": "telemetry",
    }


# ===================================================================================
#                               TRANSFORMATION STATS (from events.jsonl)
# ===================================================================================
def load_transformation_stats(run_id: str, telemetry_dir: str) -> dict | None:
    events_path = os.path.join(telemetry_dir, "events.jsonl")
    if not os.path.exists(events_path):
        return None
    try:
        events = read_jsonl(events_path)
    except Exception:
        return None

    run_events = [e for e in events if e.get("run_id") == run_id]
    transform_event = next(
        (
            e
            for e in run_events
            if e.get("event") == "transformation_finished" and e.get("task_counts")
        ),
        None,
    )
    if not transform_event:
        return None

    total_input = sum(c.get("before") or 0 for c in transform_event["task_counts"].values())
    total_output = sum(c.get("after") or 0 for c in transform_event["task_counts"].values())

    started_event = next((e for e in run_events if e.get("event") == "run_started"), None)
    finished_event = next((e for e in run_events if e.get("event") == "run_finished"), None)
    duration_seconds = None
    if started_event and finished_event:
        try:
            ts_start = datetime.fromisoformat(started_event["timestamp"]).timestamp()
            ts_end = datetime.fromisoformat(finished_event["timestamp"]).timestamp()
            duration_seconds = ts_end - ts_start
        except Exception:
            pass

    return {
        "totalInput": total_input,
        "totalOutput": total_output,
        "totalFiltered": total_input - total_output,
        "durationSeconds": duration_seconds,
        "source": "telemetry",
    }


# ===================================================================================
#                               TOKEN USAGE
# ===================================================================================
def load_run_time_context(run_id: str, telemetry_dir: str) -> dict:
    events_path = os.path.join(telemetry_dir, "events.jsonl")
    run_started_at = None
    run_finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        events = read_jsonl(events_path)
        run_events = [e for e in events if e.get("run_id") == run_id]
        started = next((e for e in run_events if e.get("event") == "run_started"), None)
        finished = next((e for e in run_events if e.get("event") == "run_finished"), None)
        if started and started.get("timestamp"):
            run_started_at = started["timestamp"][:19]
        if finished and finished.get("timestamp"):
            run_finished_at = finished["timestamp"][:19]
    except Exception:
        pass

    start_ms = (
        datetime.fromisoformat(run_started_at).replace(tzinfo=timezone.utc).timestamp() * 1000
        if run_started_at
        else None
    )
    end_ms = datetime.fromisoformat(run_finished_at).replace(tzinfo=timezone.utc).timestamp() * 1000
    duration_ms = (end_ms - start_ms) if start_ms is not None else 0

    return {
        "runStartedAt": run_started_at,
        "runFinishedAt": run_finished_at,
        "bucketSizeSeconds": adaptive_bucket_secs(duration_ms),
    }


def _lookup_rate(rates: dict, provider: str | None, model_id: str | None) -> dict | None:
    if not provider or not model_id:
        return None
    provider_entry = rates.get("providers", {}).get(provider.lower())
    if not provider_entry:
        return None
    models = provider_entry.get("models", {})
    if model_id in models:
        return {
            "rate": models[model_id],
            "updated_at": provider_entry.get("updated_at"),
            "description": provider_entry.get("description"),
        }
    lower = model_id.lower()
    for key, rate in models.items():
        if lower.startswith(key.lower()):
            return {
                "rate": rate,
                "updated_at": provider_entry.get("updated_at"),
                "description": provider_entry.get("description"),
            }
    return None


# Path to the bundled rates.json (lives in studio/src/data/ in the source tree).
_RATES_JSON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "studio", "src", "data", "rates.json")
)


def load_token_usage(run_id: str, build_id: str | None, telemetry_dir: str) -> dict | None:
    traces_path = os.path.join(telemetry_dir, "traces.jsonl")
    if not os.path.exists(traces_path):
        return None
    try:
        spans = read_jsonl(traces_path)
    except Exception:
        return None

    run_spans = [
        s for s in spans if s.get("span_name") == "dgt.llm_call" and s.get("run_id") == run_id
    ]
    if not run_spans:
        return None

    multi_task_run = bool(
        build_id
        and build_id != "exp"
        and any(
            s.get("span_name") == "dgt.llm_call"
            and s.get("build_id") == build_id
            and s.get("run_id") != run_id
            for s in spans
        )
    )

    rates_file = None
    try:
        with open(_RATES_JSON, encoding="utf-8") as f:
            rates_file = json.load(f)
    except Exception:
        pass

    prompt_tokens = 0
    completion_tokens = 0
    estimated_cost = 0.0
    rated_tokens = 0
    has_rate = False
    rate_provider = None
    rate_updated_at = None
    rate_description = None

    for s in run_spans:
        pt = s.get("prompt_tokens") or 0
        ct = s.get("completion_tokens") or 0
        prompt_tokens += pt
        completion_tokens += ct
        if rates_file:
            match = _lookup_rate(rates_file, s.get("provider"), s.get("model_id"))
            if match:
                estimated_cost += pt * match["rate"]["input"] + ct * match["rate"]["output"]
                rated_tokens += pt + ct
                has_rate = True
                rate_provider = rate_provider or s.get("provider")
                rate_updated_at = rate_updated_at or match["updated_at"]
                rate_description = rate_description or match["description"]

    avg_cost_per_token = (estimated_cost / rated_tokens) if has_rate and rated_tokens > 0 else None

    ctx = load_run_time_context(run_id, telemetry_dir)
    bucket_secs = ctx["bucketSizeSeconds"]

    bucket_map: dict[str, dict] = {}
    for s in run_spans:
        if not s.get("start_time"):
            continue
        key = to_bucket(s["start_time"], bucket_secs)
        if key in bucket_map:
            bucket_map[key]["prompt_tokens"] += s.get("prompt_tokens") or 0
            bucket_map[key]["completion_tokens"] += s.get("completion_tokens") or 0
        else:
            bucket_map[key] = {
                "timestamp": key,
                "prompt_tokens": s.get("prompt_tokens") or 0,
                "completion_tokens": s.get("completion_tokens") or 0,
            }

    series = sorted(bucket_map.values(), key=lambda b: b["timestamp"])

    result: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "series": series,
        "multi_task_run": multi_task_run,
    }
    if has_rate:
        result["estimated_cost"] = estimated_cost
        result["avg_cost_per_token"] = avg_cost_per_token
        result["rate_provider"] = rate_provider
        result["rate_updated_at"] = rate_updated_at
        result["rate_description"] = rate_description

    return result
