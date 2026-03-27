// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import React from 'react';
import { Tile, Tooltip } from '@carbon/react';
import { Information } from '@carbon/icons-react';
import { LineChart, StackedBarChart } from '@carbon/charts-react';
import '@carbon/charts-react/styles.css';
import {
  ScaleTypes,
  ChartTabularData,
  LineChartOptions,
  StackedBarChartOptions,
} from '@carbon/charts';

import {
  DataPoint,
  GenerationStats,
  TransformationStats,
  TokenUsage,
} from '@/types/custom';
import { useTheme } from '@/src/common/state/theme';
import TokenMetrics from './TokenMetrics';

import classes from './StatsTab.module.scss';

// ===================================================================================
//                               HELPERS
// ===================================================================================
function formatCost(usd: number): string {
  if (usd >= 1) return `$${usd.toFixed(2)}`;
  if (usd >= 0.01) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(4)}`;
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function StatTile({
  label,
  value,
  displayValue,
  tooltip,
  tooltipAlign = 'bottom',
  formatter = formatCount,
}: {
  label: string;
  value?: number;
  displayValue?: string;
  tooltip?: string;
  tooltipAlign?: React.ComponentProps<typeof Tooltip>['align'];
  formatter?: (n: number) => string;
}) {
  const rendered = displayValue ?? (value != null ? formatter(value) : '-');
  return (
    <Tile className={classes.statTile}>
      <div className={classes.statLabel}>
        {label}
        {tooltip && (
          <Tooltip
            align={tooltipAlign}
            label={tooltip}
            className={classes.tooltipIcon}
          >
            <button type="button" className={classes.tooltipButton}>
              <Information size={14} />
            </button>
          </Tooltip>
        )}
      </div>
      <div className={classes.statValue}>{rendered}</div>
    </Tile>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return <h5 className={classes.sectionHeading}>{children}</h5>;
}

// ===================================================================================
//                               MAIN COMPONENT
// ===================================================================================
export default function StatsTab({
  datapoints,
  tokenUsage,
  generationStats,
  transformationStats,
  isRunning,
}: {
  datapoints: {
    intermediate: DataPoint[];
    postprocessed: { [key: string]: DataPoint[] };
    final: DataPoint[];
    formatted: DataPoint[];
  };
  tokenUsage: TokenUsage | null;
  generationStats: GenerationStats | null;
  transformationStats: TransformationStats | null;
  isRunning: boolean;
}) {
  const { theme } = useTheme();
  // ---------------------------------------------------------------------------
  // Transformation task path (telemetry)
  // ---------------------------------------------------------------------------
  if (transformationStats) {
    const { totalInput, totalOutput, totalFiltered, durationSeconds } =
      transformationStats;
    const acceptanceRate =
      totalInput > 0 ? (totalOutput / totalInput) * 100 : null;
    const throughput =
      durationSeconds != null && durationSeconds > 0
        ? totalInput / durationSeconds
        : null;

    return (
      <div className={classes.statsTab}>
        <SectionHeading>Transformation</SectionHeading>
        <div className={classes.statRow}>
          <StatTile
            label="Input"
            value={totalInput}
            tooltip="Total data points passed into the transformation."
            tooltipAlign="bottom-left"
          />
          <StatTile
            label="Output"
            value={totalOutput}
            tooltip="Data points remaining after transformation and postprocessing."
          />
          <StatTile
            label="Filtered"
            value={totalFiltered}
            tooltip="Data points removed by postprocessors after transformation."
          />
          <StatTile
            label="Acceptance rate"
            displayValue={
              acceptanceRate != null ? `${acceptanceRate.toFixed(1)}%` : '-'
            }
            tooltip="Percentage of input data points that survived postprocessing."
            tooltipAlign="bottom-right"
          />
        </div>
        <div className={classes.statRow}>
          <StatTile
            label="Duration"
            displayValue={
              durationSeconds != null ? `${durationSeconds.toFixed(1)}s` : '-'
            }
            tooltip="Wall-clock time from run_started to run_finished."
            tooltipAlign="bottom-left"
          />
          <StatTile
            label="Throughput"
            displayValue={
              throughput != null ? `${throughput.toFixed(1)}/s` : '-'
            }
            tooltip="Input data points processed per second."
          />
          <div className={classes.statTileSpacer} />
          <div className={classes.statTileSpacer} />
        </div>
        <SectionHeading>Token Usage</SectionHeading>
        <TokenMetrics usage={tokenUsage} />
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Generation task path (telemetry, primary)
  // ---------------------------------------------------------------------------
  if (generationStats) {
    const {
      totalGenerated,
      totalSurvived,
      lastEpoch,
      lastEpochGenerated,
      lastEpochSurvived,
      lastEpochGenerationAttempts,
      series,
      timeSeries,
    } = generationStats;
    const totalFiltered = totalGenerated - totalSurvived;
    const acceptanceRate =
      totalGenerated > 0 ? (totalSurvived / totalGenerated) * 100 : null;
    const lastEpochFiltered = lastEpochGenerated - lastEpochSurvived;
    const lastEpochAcceptance =
      lastEpochGenerated > 0
        ? (lastEpochSurvived / lastEpochGenerated) * 100
        : null;

    const multiEpoch = series.length > 1;

    // Waste estimate using telemetry token data
    const avgTokensPerPoint =
      totalGenerated > 0 && tokenUsage
        ? (tokenUsage.prompt_tokens + tokenUsage.completion_tokens) /
          totalGenerated
        : null;
    const wastedTokens =
      avgTokensPerPoint != null && totalFiltered > 0
        ? Math.round(avgTokensPerPoint * totalFiltered)
        : null;
    const wastedCost =
      wastedTokens != null && tokenUsage?.avg_cost_per_token != null
        ? avgTokensPerPoint! * totalFiltered * tokenUsage.avg_cost_per_token
        : null;

    // Chart: cumulative generated vs survived over time (adaptive buckets)
    const chartData: ChartTabularData = timeSeries.flatMap((b) => [
      { group: 'Generated', date: b.timestamp, value: b.cumGenerated },
      {
        group: 'Survived postprocessing',
        date: b.timestamp,
        value: b.cumSurvived,
      },
    ]);

    const maxCumGenerated = Math.max(...timeSeries.map((b) => b.cumGenerated));

    const chartOptions: LineChartOptions = {
      title: 'Data points over time',
      axes: {
        bottom: {
          title: 'Time',
          mapsTo: 'date',
          scaleType: ScaleTypes.TIME,
        },
        left: {
          title: 'Cumulative data points',
          mapsTo: 'value',
          scaleType: ScaleTypes.LINEAR,
          domain: [0, Math.ceil(maxCumGenerated * 1.1)],
        },
      },
      curve: 'curveStepAfter',
      theme,
      height: '300px',
      toolbar: { enabled: false },
      legend: { enabled: true },
      tooltip: { enabled: true },
    };

    // Chart: generation vs postprocessing time per epoch (stacked bar)
    const hasTimingData = series.some(
      (b) => b.generationMs > 0 || b.postprocessingMs > 0,
    );
    const epochChartData: ChartTabularData = series.flatMap((b) => [
      {
        group: 'Generation',
        key: `Epoch ${b.epoch}`,
        value: Math.round(b.generationMs / 100) / 10,
      },
      {
        group: 'Postprocessing',
        key: `Epoch ${b.epoch}`,
        value: Math.round(b.postprocessingMs / 100) / 10,
      },
    ]);
    const maxEpochSecs = Math.max(
      ...series.map((b) => (b.generationMs + b.postprocessingMs) / 1000),
    );
    const epochChartOptions: StackedBarChartOptions = {
      title: 'Time per epoch',
      axes: {
        bottom: {
          title: 'Epoch',
          mapsTo: 'key',
          scaleType: ScaleTypes.LABELS,
        },
        left: {
          title: 'Seconds',
          mapsTo: 'value',
          scaleType: ScaleTypes.LINEAR,
          domain: [0, Math.ceil(maxEpochSecs * 1.1)],
        },
      },
      theme,
      height: '300px',
      toolbar: { enabled: false },
      legend: { enabled: true },
      tooltip: { enabled: true },
    };

    return (
      <div className={classes.statsTab}>
        <SectionHeading>Data Points (lifetime totals)</SectionHeading>
        <div className={classes.statRow}>
          <StatTile
            label="Generated"
            value={totalGenerated}
            tooltip="Total data points generated across all epochs, before postprocessing."
          />
          <StatTile
            label="Filtered"
            value={totalFiltered}
            tooltip="Total data points removed by postprocessors across all epochs."
          />
          <StatTile
            label="Survived"
            value={totalSurvived}
            tooltip="Total data points that passed all postprocessing steps across all epochs."
          />
          <StatTile
            label="Acceptance rate"
            displayValue={
              acceptanceRate != null ? `${acceptanceRate.toFixed(1)}%` : '-'
            }
            tooltip="Percentage of generated data points that survived postprocessing, across all epochs."
            tooltipAlign="bottom-right"
          />
        </div>

        {multiEpoch && (
          <>
            <SectionHeading>Last Epoch (epoch {lastEpoch})</SectionHeading>
            <div className={classes.statRow}>
              <StatTile
                label="Generated"
                value={lastEpochGenerated}
                tooltip={`Data points generated in epoch ${lastEpoch}, before postprocessing.`}
              />
              <StatTile
                label="Filtered"
                value={lastEpochFiltered}
                tooltip={`Data points removed by postprocessors in epoch ${lastEpoch}.`}
              />
              <StatTile
                label="Acceptance rate"
                displayValue={
                  lastEpochAcceptance != null
                    ? `${lastEpochAcceptance.toFixed(1)}%`
                    : '-'
                }
                tooltip={`Postprocessing acceptance rate for epoch ${lastEpoch}.`}
              />
              <StatTile
                label="Generation attempts"
                value={lastEpochGenerationAttempts}
                tooltip={`Number of generator batches that ran in epoch ${lastEpoch} before postprocessing. High values indicate the generator had to work hard to produce enough candidates.`}
                tooltipAlign="bottom-right"
              />
            </div>
          </>
        )}

        {!multiEpoch && (
          <>
            <SectionHeading>Generation</SectionHeading>
            <div className={classes.statRow}>
              <StatTile
                label="Generation attempts"
                value={lastEpochGenerationAttempts}
                tooltip="Number of generator batches that ran before postprocessing. High values indicate the generator had to work hard to produce enough candidates."
              />
              <StatTile
                label="Tokens wasted on filtered"
                displayValue={
                  wastedTokens != null
                    ? formatCount(wastedTokens)
                    : totalFiltered === 0
                      ? '0'
                      : '-'
                }
                tooltip={
                  wastedTokens != null
                    ? 'Estimated tokens spent generating data points that were later filtered out. Based on average tokens per generated data point from telemetry.'
                    : tokenUsage == null
                      ? 'Requires telemetry to be enabled (DGT_TELEMETRY_DIR).'
                      : totalFiltered === 0
                        ? 'No tokens were wasted — all generated data points survived postprocessing.'
                        : 'No rate found for the provider or model. Add an entry to rates.json to enable this estimate.'
                }
              />
              <StatTile
                label="Cost wasted on filtered"
                displayValue={
                  wastedCost != null
                    ? formatCost(wastedCost)
                    : totalFiltered === 0
                      ? '$0'
                      : '-'
                }
                tooltip={
                  wastedCost != null
                    ? 'Estimated cost of tokens spent on filtered data points.'
                    : tokenUsage == null
                      ? 'Requires telemetry to be enabled (DGT_TELEMETRY_DIR).'
                      : totalFiltered === 0
                        ? 'No cost wasted — all generated data points survived postprocessing.'
                        : 'No rate found for the provider or model. Add an entry to rates.json to enable this estimate.'
                }
                tooltipAlign="bottom-right"
              />
              <div className={classes.statTileSpacer} />
            </div>
          </>
        )}

        {timeSeries.length > 0 && (
          <div className={classes.chart}>
            <LineChart data={chartData} options={chartOptions} />
          </div>
        )}

        {hasTimingData && (
          <div className={classes.chart}>
            <StackedBarChart
              data={epochChartData}
              options={epochChartOptions}
            />
          </div>
        )}

        <SectionHeading>Token Usage</SectionHeading>
        <TokenMetrics usage={tokenUsage} />
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // File-based fallback (no telemetry)
  // ---------------------------------------------------------------------------
  const lastPostprocKey = Object.keys(datapoints.postprocessed)
    .map(Number)
    .sort((a, b) => a - b)
    .at(-1);
  const lastPostproc =
    lastPostprocKey != null
      ? datapoints.postprocessed[String(lastPostprocKey)]
      : null;

  let outputCount: number;
  let outputSource: 'final' | 'postproc' | 'intermediate';
  if (!isRunning && datapoints.final.length > 0) {
    outputCount = datapoints.final.length;
    outputSource = 'final';
  } else if (lastPostproc != null) {
    outputCount = lastPostproc.length;
    outputSource = 'postproc';
  } else {
    outputCount = datapoints.intermediate.length;
    outputSource = 'intermediate';
  }

  const epoch1Count = datapoints.intermediate.length;

  let filtered: number | null;
  let acceptanceRate: number | null;
  if (epoch1Count === 0) {
    filtered = null;
    acceptanceRate = null;
  } else if (outputSource === 'intermediate') {
    filtered = 0;
    acceptanceRate = 100;
  } else if (outputCount >= epoch1Count) {
    filtered = 0;
    acceptanceRate = 100;
  } else {
    filtered = epoch1Count - outputCount;
    acceptanceRate = (outputCount / epoch1Count) * 100;
  }

  const avgTokensPerPoint =
    epoch1Count > 0 && tokenUsage
      ? (tokenUsage.prompt_tokens + tokenUsage.completion_tokens) / epoch1Count
      : null;
  const wastedTokens =
    avgTokensPerPoint != null && filtered != null && filtered > 0
      ? Math.round(avgTokensPerPoint * filtered)
      : null;
  const wastedCost =
    avgTokensPerPoint != null &&
    filtered != null &&
    filtered > 0 &&
    tokenUsage?.avg_cost_per_token != null
      ? avgTokensPerPoint * filtered * tokenUsage.avg_cost_per_token
      : null;

  const outputTooltip = isRunning
    ? outputSource === 'postproc'
      ? `Current retained pool from postproc_data_${lastPostprocKey}.jsonl. Updates every few seconds.`
      : 'Epoch 1 raw generation. No postprocessing has run yet.'
    : outputSource === 'final'
      ? 'Final output written at run completion (final_data.jsonl).'
      : outputSource === 'postproc'
        ? `Latest retained pool from postproc_data_${lastPostprocKey}.jsonl. final_data.jsonl was not found.`
        : 'Epoch 1 raw generation. No postprocessing ran for this task.';

  const filteredTooltip =
    filtered == null
      ? 'Filtered count is not available. No generation data was found.'
      : filtered === 0 && outputSource === 'intermediate'
        ? 'No postprocessing ran for this task.'
        : filtered === 0
          ? 'Postprocessing ran but did not reduce the candidate pool.'
          : 'Candidates from epoch 1 that did not survive postprocessing. Only reliable for single-epoch runs. Enable DGT_TELEMETRY_DIR for exact multi-epoch counts.';

  const acceptanceTooltip =
    acceptanceRate == null
      ? 'Acceptance rate is not available. No generation data was found.'
      : acceptanceRate === 100 && outputSource === 'intermediate'
        ? 'No postprocessing ran for this task.'
        : acceptanceRate === 100
          ? 'Postprocessing ran but did not reduce the candidate pool.'
          : 'Percentage of epoch 1 candidates that survived postprocessing. Only reliable for single-epoch runs. Enable DGT_TELEMETRY_DIR for exact multi-epoch counts.';

  return (
    <div className={classes.statsTab}>
      <SectionHeading>Data Points</SectionHeading>
      <div className={classes.statRow}>
        <StatTile
          label="Output"
          value={outputCount}
          tooltip={outputTooltip}
          tooltipAlign="bottom-left"
        />
        <StatTile
          label="Filtered"
          displayValue={filtered != null ? formatCount(filtered) : '-'}
          tooltip={filteredTooltip}
        />
        <StatTile
          label="Acceptance rate"
          displayValue={
            acceptanceRate != null ? `${acceptanceRate.toFixed(1)}%` : '-'
          }
          tooltip={acceptanceTooltip}
        />
        <StatTile
          label="Tokens wasted on filtered"
          displayValue={
            wastedTokens != null
              ? formatCount(wastedTokens)
              : filtered === 0
                ? '0'
                : '-'
          }
          tooltip={
            wastedTokens != null
              ? 'Estimated tokens spent generating data points that were later filtered out. Only reliable for single-epoch runs.'
              : tokenUsage == null
                ? 'Requires telemetry to be enabled. Set DGT_TELEMETRY_DIR to point to your telemetry directory.'
                : filtered == null || filtered === 0
                  ? 'No tokens were wasted on filtered data points.'
                  : 'No rate found for the provider or model used in this run.'
          }
          tooltipAlign="bottom-right"
        />
      </div>
      <div className={classes.statRow}>
        <StatTile
          label="Cost wasted on filtered"
          displayValue={
            wastedCost != null
              ? formatCost(wastedCost)
              : filtered === 0
                ? '$0'
                : '-'
          }
          tooltip={
            wastedCost != null
              ? 'Estimated cost of tokens spent on filtered data points. Only reliable for single-epoch runs.'
              : tokenUsage == null
                ? 'Requires telemetry to be enabled. Set DGT_TELEMETRY_DIR to point to your telemetry directory.'
                : filtered == null || filtered === 0
                  ? 'No cost was wasted on filtered data points.'
                  : 'No rate found for the provider or model used in this run.'
          }
          tooltipAlign="bottom-left"
        />
        <div className={classes.statTileSpacer} />
        <div className={classes.statTileSpacer} />
        <div className={classes.statTileSpacer} />
      </div>

      <SectionHeading>Token Usage</SectionHeading>
      <TokenMetrics usage={tokenUsage} />
    </div>
  );
}
