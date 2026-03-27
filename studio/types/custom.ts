// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

// ===================================================================================
//                               NOTIFICATION
// ===================================================================================
export interface Notification {
  title: string;
  subtitle: string;
  kind:
    | 'error'
    | 'info'
    | 'info-square'
    | 'success'
    | 'warning'
    | 'warning-alt';
  caption?: string;
  timeout?: number;
  type?: 'Toast' | 'Inline' | 'Actionable';
  onCloseButtonClick?: () => {};
  onActionButtonClick?: () => {};
}

// ===================================================================================
//                              DATA POINT
// ===================================================================================
export type DataPoint = { [key: string]: any };

// ===================================================================================
//                               RUN INFORMATION CARD
// ===================================================================================
export interface RunInformationCard {
  name: string;
  path: string;
  status: string;
  startTime: Date;
  duration: string;
}

// ===================================================================================
//                              FMS-DGT RESOURCES
// ===================================================================================
export interface DGT_TASK {
  data_builder: string;
  task_name: string;
  task_description?: string;
  seed_examples?: any[];
  [key: string]: any;
}

interface TaskSpecifications {
  task_init: DGT_TASK;
  task_kwargs: { [key: string]: any };
}

interface DGT_DATABUILDER_BLOCK {
  name: string;
  type: string;
  base_url?: string;
  model_id_or_path?: string;
  temperature?: number;
  max_new_tokens?: number;
  min_new_tokens?: number;
}

interface DGT_DATABUILDER_POSTPROCESSOR {
  name: string;
  [key: string]: any;
}

interface DGT_DATA_BUILDER {
  name: string;
  blocks: DGT_DATABUILDER_BLOCK[];
  postprocessors: DGT_DATABUILDER_POSTPROCESSOR[];
}

export interface DGT_TASK_CARD {
  task_name: string;
  databuilder_name: string;
  task_spec: TaskSpecifications;
  databuilder_spec: DGT_DATA_BUILDER;
  build_id?: string;
  run_id?: string;
  process_id?: string | number;
}

export interface DGT_TASK_RESULT {
  PID: string | number;
  status: 'running' | 'completed' | 'errored';
  start_time: number;
  end_time?: number;
  metrics: { [key: string]: any };
}

// ===================================================================================
//                              TELEMETRY — EVENTS
// ===================================================================================
export interface DGT_EVENT {
  event: string;
  run_id: string;
  build_id: string;
  timestamp: string; // ISO 8601
  [key: string]: any;
}

export interface EpochStatsBucket {
  epoch: number;
  generated: number; // before postprocessing
  survived: number; // after postprocessing
  generationAttempts: number;
  generationMs: number; // time spent in generation (epoch - postprocessing), ms
  postprocessingMs: number; // time spent in postprocessing, ms
}

export interface DataPointsTimeSeriesBucket {
  timestamp: string; // ISO 10-second bucket, e.g. "2026-03-25T10:04:10"
  cumGenerated: number; // cumulative generated up to and including this epoch
  cumSurvived: number; // cumulative survived up to and including this epoch
}

export interface GenerationStats {
  // Lifetime totals across all epochs
  totalGenerated: number;
  totalSurvived: number;
  // Last epoch detail
  lastEpoch: number;
  lastEpochGenerated: number;
  lastEpochSurvived: number;
  lastEpochGenerationAttempts: number;
  // Per-epoch buckets for stat tiles
  series: EpochStatsBucket[];
  // Time-series for charting: one entry per postprocessing_finished event, cumulative counts
  timeSeries: DataPointsTimeSeriesBucket[];
  source: 'telemetry'; // always telemetry when present — file-based has no GenerationStats
}

export interface TransformationStats {
  // Per-task counts from transformation_finished event
  totalInput: number; // sum of before counts across all tasks
  totalOutput: number; // sum of after counts across all tasks
  totalFiltered: number; // totalInput - totalOutput (removed by postprocessors)
  durationSeconds: number | null; // run_finished - run_started, null if not available
  source: 'telemetry';
}

// ===================================================================================
//                              TELEMETRY — SPANS
// ===================================================================================
export interface DGT_LLM_SPAN {
  span_name: string;
  run_id: string;
  build_id: string;
  start_time: string; // ISO 8601
  end_time: string; // ISO 8601
  prompt_tokens: number;
  completion_tokens: number;
  provider?: string;
  model_id?: string;
  duration_ms?: number;
}

export interface TokenUsageBucket {
  timestamp: string; // ISO minute bucket e.g. "2026-03-25T10:04"
  prompt_tokens: number;
  completion_tokens: number;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  series: TokenUsageBucket[];
  multi_task_run: boolean; // true when >1 distinct run_id shares the same build_id
  estimated_cost?: number; // USD, absent when no rate found for provider+model
  avg_cost_per_token?: number; // USD per token (weighted avg across all spans with a rate), used for waste estimate
  rate_provider?: string; // Provider name (e.g. "anthropic"), shown in cost tooltip
  rate_updated_at?: string; // ISO date from the provider's entry in rates.json
  rate_description?: string; // Provider description from rates.json, shown in cost tooltip
}
