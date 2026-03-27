// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';

import {
  DataPointsTimeSeriesBucket,
  DGT_EVENT,
  DGT_LLM_SPAN,
  DGT_TASK_CARD,
  DGT_TASK_RESULT,
  EpochStatsBucket,
  GenerationStats,
  RunInformationCard,
  TokenUsage,
  TokenUsageBucket,
  TransformationStats,
} from '@/types/custom';
import { castDurationToString } from '@/src/common/utilities/time';

// ===================================================================================
//                               JSONL PARSING
// ===================================================================================
export function parseJsonl<T>(content: string): T[] {
  return JSON.parse(
    '[' +
      content
        .split(/\n/)
        .filter((l) => l.trim())
        .join(',') +
      ']',
  );
}

// ===================================================================================
//                               RUN DISCOVERY
// ===================================================================================
export function walk(
  directory: string,
  name: string,
  root: string,
): RunInformationCard[] {
  let runs: RunInformationCard[] = [];

  fs.readdirSync(directory).forEach((filename) => {
    const filePath = `${directory}/${filename}`;
    const stat = fs.statSync(filePath);

    if (stat && stat.isDirectory()) {
      const filenames = fs.readdirSync(filePath);

      if (
        filenames.includes('task_card.jsonl') &&
        filenames.includes('task_results.jsonl')
      ) {
        let startTime: Date = fs.statSync(`${filePath}/task_card.jsonl`).mtime;
        let duration = '-';
        let status = 'running';

        try {
          const resultsData = parseJsonl<DGT_TASK_RESULT>(
            fs.readFileSync(`${filePath}/task_results.jsonl`, 'utf8'),
          );
          const last = resultsData[resultsData.length - 1];

          startTime = new Date(last.start_time * 1000);
          const endTime = last.end_time
            ? new Date(last.end_time * 1000)
            : new Date();
          const [days, hours, mins, secs] = castDurationToString(
            (endTime.getTime() - startTime.getTime()) / 1000,
          );
          duration = `${days ? days + ' days ' : ''}${
            hours ? hours + ' hours ' : ''
          }${mins ? mins + ' mins ' : ''}${secs} sec`;
          status = last.status;
        } catch {
          try {
            const [days, hours, mins, secs] = castDurationToString(
              fs.statSync(`${filePath}/task_results.jsonl`).mtime.getTime() -
                startTime.getTime(),
            );
            duration = `${days ? days + ' days ' : ''}${
              hours ? hours + ' hours ' : ''
            }${mins ? mins + ' mins ' : ''}${secs} sec`;
          } catch {}
        }

        runs.push({
          name: name ? `${name}/${filename}` : filename,
          path: `${root}${name}/${filename}`,
          status,
          startTime,
          duration,
        });
      } else {
        runs = runs.concat(walk(filePath, `${name}/${filename}`, root));
      }
    }
  });

  return runs;
}

// ===================================================================================
//                               RUN DETAIL LOADERS
// ===================================================================================
export function loadTaskCards(path: string): DGT_TASK_CARD[] {
  return parseJsonl<DGT_TASK_CARD>(
    fs.readFileSync(`${path}/task_card.jsonl`, 'utf8'),
  );
}

export function loadLog(path: string): string | undefined {
  // Prefer logs.jsonl (public-repo LogDatastoreHandler format).
  try {
    const entries = fs
      .readFileSync(`${path}/logs.jsonl`, 'utf8')
      .split('\n')
      .filter((l) => l.trim())
      .map((l) => {
        const e = JSON.parse(l);
        return `${e.timestamp} ${e.level.padEnd(8)} [${e.logger}:${e.lineno}] ${
          e.message
        }`;
      });
    if (entries.length > 0) return entries.join('\n');
  } catch {}

  // Fall back to plain-text .log files (older internal-repo format, sorted by mtime for resumed runs).
  try {
    const logsDir = `${path}/logs`;
    const logFiles = fs
      .readdirSync(logsDir)
      .filter((f) => f.endsWith('.log'))
      .sort(
        (a, b) =>
          fs.statSync(`${logsDir}/${a}`).mtimeMs -
          fs.statSync(`${logsDir}/${b}`).mtimeMs,
      );
    if (logFiles.length > 0) {
      return logFiles
        .map((f) => fs.readFileSync(`${logsDir}/${f}`, 'utf8'))
        .join('\n');
    }
  } catch {}

  return undefined;
}

export function loadResults(path: string): { [key: string]: DGT_TASK_RESULT } {
  const taskResults = parseJsonl<DGT_TASK_RESULT>(
    fs.readFileSync(`${path}/task_results.jsonl`, 'utf8'),
  );
  return Object.fromEntries(taskResults.map((r) => [String(r.PID), r]));
}

export function loadDataPoints(path: string): {
  intermediate: any[];
  postprocessed: { [key: string]: any[] };
  final: any[];
  formatted: any[];
} {
  const data = {
    intermediate: [],
    postprocessed: {},
    final: [],
    formatted: [],
  } as {
    intermediate: any[];
    postprocessed: { [key: string]: any[] };
    final: any[];
    formatted: any[];
  };

  const parseJsonlFile = (file: string) =>
    parseJsonl(fs.readFileSync(`${path}/${file}`, 'utf8'));

  fs.readdirSync(path).forEach((filename) => {
    try {
      if (filename === 'data.jsonl') {
        data.intermediate = parseJsonlFile(filename);
      } else if (filename.startsWith('postproc_data_')) {
        data.postprocessed[filename.slice(14, -6)] = parseJsonlFile(filename);
      } else if (filename === 'final_data.jsonl') {
        data.final = parseJsonlFile(filename);
      } else if (filename === 'formatted_data.jsonl') {
        data.formatted = parseJsonlFile(filename);
      }
    } catch {}
  });

  return data;
}

export function resolvePid(
  taskCard: DGT_TASK_CARD,
  results: { [key: string]: DGT_TASK_RESULT },
): string {
  if (taskCard.process_id != null) return String(taskCard.process_id);
  return String(
    Object.values(results)
      .map((r) => [r.PID, r.start_time] as [string | number, number])
      // @ts-ignore — toSorted not in all TS lib versions
      .toSorted(
        (a: [string | number, number], b: [string | number, number]) =>
          b[1] - a[1],
      )[0][0],
  );
}

// ===================================================================================
//                               EPOCH TIMINGS (from traces.jsonl)
// ===================================================================================
function loadEpochTimings(
  runId: string,
  telemetryDir: string,
): Map<number, { generationMs: number; postprocessingMs: number }> {
  const result = new Map<
    number,
    { generationMs: number; postprocessingMs: number }
  >();
  const tracesPath = `${telemetryDir}/traces.jsonl`;
  if (!fs.existsSync(tracesPath)) return result;

  let spans: any[];
  try {
    spans = parseJsonl(fs.readFileSync(tracesPath, 'utf8'));
  } catch {
    return result;
  }

  const runSpans = spans.filter((s) => s.run_id === runId && s.epoch != null);

  for (const s of runSpans) {
    const epoch: number = s.epoch;
    const ms: number = s.duration_ms ?? 0;
    const existing = result.get(epoch) ?? {
      generationMs: 0,
      postprocessingMs: 0,
    };

    if (s.span_name === 'dgt.epoch') {
      result.set(epoch, { ...existing, generationMs: ms });
    } else if (s.span_name === 'dgt.postprocessing') {
      result.set(epoch, { ...existing, postprocessingMs: ms });
    }
  }

  // Subtract postprocessing from epoch to get pure generation time.
  for (const [epoch, timing] of result.entries()) {
    result.set(epoch, {
      ...timing,
      generationMs: Math.max(0, timing.generationMs - timing.postprocessingMs),
    });
  }

  return result;
}

// ===================================================================================
//                               GENERATION STATS (from events.jsonl)
// ===================================================================================
export function loadGenerationStats(
  runId: string,
  telemetryDir: string,
): GenerationStats | null {
  const eventsPath = `${telemetryDir}/events.jsonl`;
  if (!fs.existsSync(eventsPath)) return null;

  let events: DGT_EVENT[];
  try {
    events = parseJsonl<DGT_EVENT>(fs.readFileSync(eventsPath, 'utf8'));
  } catch {
    return null;
  }

  const runEvents = events.filter((e) => e.run_id === runId);

  // Collect per-epoch data from postprocessing_finished and epoch_finished events.
  // Key: epoch number. Values merged from both event types.
  const epochMap = new Map<
    number,
    { generated: number; survived: number; generationAttempts: number }
  >();

  for (const e of runEvents) {
    if (
      e.event === 'postprocessing_finished' &&
      e.epoch != null &&
      e.task_counts
    ) {
      let generated = 0;
      let survived = 0;
      for (const counts of Object.values(e.task_counts) as Array<{
        before: number;
        after: number;
      }>) {
        generated += counts.before ?? 0;
        survived += counts.after ?? 0;
      }
      const existing = epochMap.get(e.epoch) ?? {
        generated: 0,
        survived: 0,
        generationAttempts: 0,
      };
      epochMap.set(e.epoch, { ...existing, generated, survived });
    }

    if (e.event === 'epoch_finished' && e.epoch != null) {
      const attempts = e.generation_attempts ?? 0;
      const existing = epochMap.get(e.epoch) ?? {
        generated: 0,
        survived: 0,
        generationAttempts: 0,
      };
      epochMap.set(e.epoch, { ...existing, generationAttempts: attempts });
    }
  }

  if (epochMap.size === 0) return null;

  const epochTimings = loadEpochTimings(runId, telemetryDir);

  const series: EpochStatsBucket[] = Array.from(epochMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([epoch, data]) => ({
      epoch,
      ...data,
      generationMs: epochTimings.get(epoch)?.generationMs ?? 0,
      postprocessingMs: epochTimings.get(epoch)?.postprocessingMs ?? 0,
    }));

  const totalGenerated = series.reduce((s, b) => s + b.generated, 0);
  const totalSurvived = series.reduce((s, b) => s + b.survived, 0);
  const last = series[series.length - 1];

  // Build time-series using adaptive bucket size anchored to run_started.
  // Derive timing directly from the already-parsed runEvents to avoid a second file read.
  const startedEvent = runEvents.find((e) => e.event === 'run_started');
  const finishedEvent = runEvents.find((e) => e.event === 'run_finished');
  const runStartedAt = startedEvent?.timestamp?.slice(0, 19) ?? null;
  const runFinishedAt =
    finishedEvent?.timestamp?.slice(0, 19) ??
    new Date().toISOString().slice(0, 19);
  const startMs = runStartedAt ? new Date(runStartedAt + 'Z').getTime() : null;
  const endMs = new Date(runFinishedAt + 'Z').getTime();
  const bucketSizeSeconds = adaptiveBucketSecs(
    startMs != null ? endMs - startMs : 0,
  );

  const postprocEvents = runEvents
    .filter(
      (e) =>
        e.event === 'postprocessing_finished' && e.timestamp && e.task_counts,
    )
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  // Zero baseline: one bucket before run_started so the TIME axis always has a range.
  const zeroAnchor = runStartedAt
    ? shiftBucket(toBucket(runStartedAt, bucketSizeSeconds), -bucketSizeSeconds)
    : null;

  let cumGenerated = 0;
  let cumSurvived = 0;
  const timeSeries: DataPointsTimeSeriesBucket[] = [
    ...(zeroAnchor
      ? [{ timestamp: zeroAnchor, cumGenerated: 0, cumSurvived: 0 }]
      : []),
    ...postprocEvents.map((e) => {
      for (const counts of Object.values(e.task_counts) as Array<{
        before: number;
        after: number;
      }>) {
        cumGenerated += counts.before ?? 0;
        cumSurvived += counts.after ?? 0;
      }
      return {
        timestamp: toBucket(e.timestamp, bucketSizeSeconds),
        cumGenerated,
        cumSurvived,
      };
    }),
  ];

  return {
    totalGenerated,
    totalSurvived,
    lastEpoch: last.epoch,
    lastEpochGenerated: last.generated,
    lastEpochSurvived: last.survived,
    lastEpochGenerationAttempts: last.generationAttempts,
    series,
    timeSeries,
    source: 'telemetry',
  };
}

// ===================================================================================
//                               TRANSFORMATION STATS (from events.jsonl)
// ===================================================================================
export function loadTransformationStats(
  runId: string,
  telemetryDir: string,
): TransformationStats | null {
  const eventsPath = `${telemetryDir}/events.jsonl`;
  if (!fs.existsSync(eventsPath)) return null;

  let events: DGT_EVENT[];
  try {
    events = parseJsonl<DGT_EVENT>(fs.readFileSync(eventsPath, 'utf8'));
  } catch {
    return null;
  }

  const runEvents = events.filter((e) => e.run_id === runId);
  const transformEvent = runEvents.find(
    (e) => e.event === 'transformation_finished',
  );
  if (!transformEvent?.task_counts) return null;

  let totalInput = 0;
  let totalOutput = 0;
  for (const counts of Object.values(transformEvent.task_counts) as Array<{
    before: number;
    after: number;
  }>) {
    totalInput += counts.before ?? 0;
    totalOutput += counts.after ?? 0;
  }

  const startedEvent = runEvents.find((e) => e.event === 'run_started');
  const finishedEvent = runEvents.find((e) => e.event === 'run_finished');
  let durationSeconds: number | null = null;
  if (startedEvent?.timestamp && finishedEvent?.timestamp) {
    durationSeconds =
      (new Date(finishedEvent.timestamp).getTime() -
        new Date(startedEvent.timestamp).getTime()) /
      1000;
  }

  return {
    totalInput,
    totalOutput,
    totalFiltered: totalInput - totalOutput,
    durationSeconds,
    source: 'telemetry',
  };
}

// ===================================================================================
//                               TOKEN USAGE
// ===================================================================================

// Truncates an ISO timestamp to a bucket of `bucketSecs` seconds.
// e.g. bucketSecs=10: "2026-03-25T10:04:32.123Z" → "2026-03-25T10:04:30"
//      bucketSecs=60: "2026-03-25T10:04:32.123Z" → "2026-03-25T10:04:00"
function toBucket(isoString: string, bucketSecs: number): string {
  const base = isoString.slice(0, 19); // "2026-03-25T10:04:32"
  const d = new Date(base + 'Z');
  const snapped =
    Math.floor(d.getTime() / (bucketSecs * 1000)) * bucketSecs * 1000;
  return new Date(snapped).toISOString().slice(0, 19);
}

// Shifts an ISO datetime string by offsetSeconds.
function shiftBucket(isoString: string, offsetSeconds: number): string {
  const d = new Date(isoString.slice(0, 19) + 'Z');
  d.setTime(d.getTime() + offsetSeconds * 1000);
  return d.toISOString().slice(0, 19);
}

// Returns bucket size in seconds based on run duration.
function adaptiveBucketSecs(durationMs: number): number {
  const mins = durationMs / 60000;
  if (mins < 1) return 5;
  if (mins < 10) return 15;
  if (mins < 60) return 60;
  return 300;
}

// Reads events.jsonl once and returns run timing context.
// runFinishedAt is either the run_finished timestamp or the current time for active runs.
export function loadRunTimeContext(
  runId: string,
  telemetryDir: string,
): {
  runStartedAt: string | null;
  runFinishedAt: string;
  bucketSizeSeconds: number;
} {
  const eventsPath = `${telemetryDir}/events.jsonl`;
  let runStartedAt: string | null = null;
  let runFinishedAt: string = new Date().toISOString().slice(0, 19);

  try {
    const events = parseJsonl<DGT_EVENT>(fs.readFileSync(eventsPath, 'utf8'));
    const runEvents = events.filter((e) => e.run_id === runId);
    const started = runEvents.find((e) => e.event === 'run_started');
    const finished = runEvents.find((e) => e.event === 'run_finished');
    if (started?.timestamp) runStartedAt = started.timestamp.slice(0, 19);
    if (finished?.timestamp) runFinishedAt = finished.timestamp.slice(0, 19);
  } catch {}

  const startMs = runStartedAt ? new Date(runStartedAt + 'Z').getTime() : null;
  const endMs = new Date(runFinishedAt + 'Z').getTime();
  const durationMs = startMs != null ? endMs - startMs : 0;

  return {
    runStartedAt,
    runFinishedAt,
    bucketSizeSeconds: adaptiveBucketSecs(durationMs),
  };
}

interface RateEntry {
  input: number;
  output: number;
}

interface ProviderEntry {
  updated_at: string;
  description: string;
  models: { [model: string]: RateEntry };
}

interface RatesFile {
  providers: { [provider: string]: ProviderEntry };
}

function lookupRate(
  rates: RatesFile,
  provider: string | undefined,
  modelId: string | undefined,
): { rate: RateEntry; updated_at: string; description: string } | null {
  if (!provider || !modelId) return null;
  const providerEntry = rates.providers[provider.toLowerCase()];
  if (!providerEntry) return null;
  const { models, updated_at, description } = providerEntry;
  // Exact match first, then case-insensitive prefix match for versioned model IDs
  const exact = models[modelId];
  if (exact) return { rate: exact, updated_at, description };
  const lower = modelId.toLowerCase();
  const prefixMatch = Object.entries(models).find(([key]) =>
    lower.startsWith(key.toLowerCase()),
  );
  return prefixMatch ? { rate: prefixMatch[1], updated_at, description } : null;
}

export function loadTokenUsage(
  runId: string,
  buildId: string | undefined,
  telemetryDir: string,
): TokenUsage | null {
  const tracesPath = `${telemetryDir}/traces.jsonl`;
  if (!fs.existsSync(tracesPath)) return null;

  let spans: DGT_LLM_SPAN[];
  try {
    spans = parseJsonl<DGT_LLM_SPAN>(fs.readFileSync(tracesPath, 'utf8'));
  } catch {
    return null;
  }

  // Filter to llm_call spans for this run
  const runSpans = spans.filter(
    (s) => s.span_name === 'dgt.llm_call' && s.run_id === runId,
  );
  if (runSpans.length === 0) return null;

  // Detect multi-task run: other run_ids sharing the same build_id
  const multi_task_run =
    !!buildId &&
    buildId !== 'exp' &&
    spans.some(
      (s) =>
        s.span_name === 'dgt.llm_call' &&
        s.build_id === buildId &&
        s.run_id !== runId,
    );

  // Load rates file (optional — cost is best-effort)
  let ratesFile: RatesFile | null = null;
  try {
    const ratesPath = `${process.cwd()}/src/data/rates.json`;
    ratesFile = JSON.parse(fs.readFileSync(ratesPath, 'utf8')) as RatesFile;
  } catch {}

  // Aggregate totals and cost
  let prompt_tokens = 0;
  let completion_tokens = 0;
  let estimated_cost = 0;
  let rated_tokens = 0; // total tokens from spans that had a rate match, for avg_cost_per_token
  let has_rate = false;
  let rate_provider: string | undefined;
  let rate_updated_at: string | undefined;
  let rate_description: string | undefined;
  for (const s of runSpans) {
    const pt = s.prompt_tokens ?? 0;
    const ct = s.completion_tokens ?? 0;
    prompt_tokens += pt;
    completion_tokens += ct;
    if (ratesFile) {
      const match = lookupRate(ratesFile, s.provider, s.model_id);
      if (match) {
        const spanCost = pt * match.rate.input + ct * match.rate.output;
        estimated_cost += spanCost;
        rated_tokens += pt + ct;
        has_rate = true;
        if (!rate_provider) rate_provider = s.provider;
        if (!rate_updated_at) rate_updated_at = match.updated_at;
        if (!rate_description) rate_description = match.description;
      }
    }
  }
  const avg_cost_per_token =
    has_rate && rated_tokens > 0 ? estimated_cost / rated_tokens : undefined;

  // Build per-bucket time-series using adaptive bucket size anchored to run_started.
  const { runStartedAt, bucketSizeSeconds } = loadRunTimeContext(
    runId,
    telemetryDir,
  );
  const bucketMap = new Map<string, TokenUsageBucket>();
  for (const s of runSpans) {
    const key = toBucket(s.start_time, bucketSizeSeconds);
    const existing = bucketMap.get(key);
    if (existing) {
      existing.prompt_tokens += s.prompt_tokens ?? 0;
      existing.completion_tokens += s.completion_tokens ?? 0;
    } else {
      bucketMap.set(key, {
        timestamp: key,
        prompt_tokens: s.prompt_tokens ?? 0,
        completion_tokens: s.completion_tokens ?? 0,
      });
    }
  }

  const sorted = Array.from(bucketMap.values()).sort((a, b) =>
    a.timestamp.localeCompare(b.timestamp),
  );

  const series: TokenUsageBucket[] = sorted;

  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: prompt_tokens + completion_tokens,
    series,
    multi_task_run,
    ...(has_rate && {
      estimated_cost,
      avg_cost_per_token,
      rate_provider,
      rate_updated_at,
      rate_description,
    }),
  };
}
