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
