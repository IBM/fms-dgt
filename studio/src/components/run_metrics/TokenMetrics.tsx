// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import React from 'react';
import { Tile, Tooltip } from '@carbon/react';
import { Information } from '@carbon/icons-react';
import { StackedBarChart } from '@carbon/charts-react';
import '@carbon/charts-react/styles.css';
import {
  ScaleTypes,
  ChartTabularData,
  StackedBarChartOptions,
} from '@carbon/charts';

import { TokenUsage } from '@/types/custom';
import { useTheme } from '@/src/common/state/theme';

import classes from './TokenMetrics.module.scss';

// ===================================================================================
//                               HELPERS
// ===================================================================================
function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(usd: number): string {
  if (usd >= 1) return `$${usd.toFixed(2)}`;
  if (usd >= 0.01) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(4)}`;
}

// ===================================================================================
//                               STAT TILE
// ===================================================================================
function StatTile({
  label,
  value,
  displayValue,
  tooltip,
  tooltipAlign = 'bottom',
  formatter = formatCount,
  muted = false,
}: {
  label: string;
  value?: number;
  displayValue?: string;
  tooltip?: string;
  tooltipAlign?: React.ComponentProps<typeof Tooltip>['align'];
  formatter?: (n: number) => string;
  muted?: boolean;
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
      <div
        className={`${classes.statValue}${
          muted ? ` ${classes.statValueMuted}` : ''
        }`}
      >
        {rendered}
      </div>
    </Tile>
  );
}

// ===================================================================================
//                               MAIN COMPONENT
// ===================================================================================
export default function TokenMetrics({ usage }: { usage: TokenUsage | null }) {
  const { theme } = useTheme();
  const noTelemetryTooltip =
    'Requires telemetry to be enabled. Set DGT_TELEMETRY_DIR to point to your telemetry directory.';

  const caveat = !usage
    ? noTelemetryTooltip
    : usage.multi_task_run
      ? 'Token counts reflect all tasks sharing this databuilder, not this task alone. Per-task attribution is not yet available.'
      : 'Token counts reflect the entire run. Prompt and completion tokens are billed at different rates by most providers.';

  const hasRate = usage?.estimated_cost != null;

  const costTooltipLabel = !usage ? (
    <div className={classes.costTooltipContent}>{noTelemetryTooltip}</div>
  ) : hasRate ? (
    <div className={classes.costTooltipContent}>
      <div>Provider: {usage.rate_provider}</div>
      <div className={classes.costTooltipSection}>{usage.rate_description}</div>
      <div className={classes.costTooltipSection}>
        Updated on: {usage.rate_updated_at}
      </div>
    </div>
  ) : (
    <div className={classes.costTooltipContent}>
      Cost estimate unavailable. The provider or model for this run is not
      present in rates.json. Add an entry under the matching provider to enable
      cost tracking.
    </div>
  );

  // Build chart data only when telemetry is available.
  // Prompt + Completion stack naturally; the top of the bar = total tokens for that bucket.
  const chartData: ChartTabularData = usage
    ? usage.series.flatMap((bucket) => [
        {
          group: 'Prompt tokens',
          date: bucket.timestamp,
          value: bucket.prompt_tokens,
        },
        {
          group: 'Completion tokens',
          date: bucket.timestamp,
          value: bucket.completion_tokens,
        },
      ])
    : [];

  // Pin Y axis to max total + 10% so the top bar segment is never clipped.
  const maxTotal = usage
    ? Math.max(
        ...usage.series.map((b) => b.prompt_tokens + b.completion_tokens),
      )
    : 0;

  const chartOptions: StackedBarChartOptions = {
    title: 'Token usage over time',
    axes: {
      bottom: {
        title: 'Time',
        mapsTo: 'date',
        scaleType: ScaleTypes.TIME,
      },
      left: {
        title: 'Tokens',
        mapsTo: 'value',
        scaleType: ScaleTypes.LINEAR,
        domain: [0, Math.ceil(maxTotal * 1.1)],
      },
    },
    theme,
    height: '300px',
    toolbar: { enabled: false },
    legend: { enabled: true },
    tooltip: { enabled: true },
  };

  return (
    <div className={classes.tokenMetrics}>
      <div className={classes.statRow}>
        <StatTile
          label="Prompt tokens"
          displayValue={usage ? formatCount(usage.prompt_tokens) : '-'}
          tooltip={caveat}
        />
        <StatTile
          label="Completion tokens"
          displayValue={usage ? formatCount(usage.completion_tokens) : '-'}
          tooltip={caveat}
        />
        <StatTile
          label="Total tokens"
          displayValue={usage ? formatCount(usage.total_tokens) : '-'}
          tooltip={caveat}
        />
        <Tile className={classes.statTile}>
          <div className={classes.statLabel}>
            Estimated cost
            <Tooltip
              align="bottom-right"
              className={classes.tooltipIcon}
              label={costTooltipLabel}
            >
              <button type="button" className={classes.tooltipButton}>
                <Information size={14} />
              </button>
            </Tooltip>
          </div>
          <div className={classes.statValue}>
            {hasRate ? formatCost(usage!.estimated_cost!) : '-'}
          </div>
        </Tile>
      </div>
      {usage && usage.series.length > 0 && (
        <div className={classes.chart}>
          <StackedBarChart data={chartData} options={chartOptions} />
        </div>
      )}
    </div>
  );
}
