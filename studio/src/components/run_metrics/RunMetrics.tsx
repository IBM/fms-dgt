// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import cx from 'classnames';
import { Tile, CodeSnippet } from '@carbon/react';

import classes from './RunMetrics.module.scss';

function Metric({ name, value }: { name: string; value: any }) {
  return (
    <Tile className={classes.metric}>
      <div className={classes.tileHeading}>{name}</div>
      <div className={classes.tileBody}>
        {Array.isArray(value) || typeof value === 'object' ? (
          <CodeSnippet
            type="multi"
            wrapText
            hideCopyButton
            maxCollapsedNumberOfRows={8}
            minExpandedNumberOfRows={16}
            maxExpandedNumberOfRows={16}
          >
            {JSON.stringify(value, null, 2)}
          </CodeSnippet>
        ) : (
          <span className={classes.metricTextValue}>
            {typeof value === 'boolean' ? (value ? 'True' : 'False') : value}
          </span>
        )}
      </div>
    </Tile>
  );
}

export default function RunMetrics({ metrics }: { metrics: {} | undefined }) {
  return (
    <>
      {metrics ? (
        <div
          className={cx(
            classes.metrics,
            Object.keys(metrics).length > 3 ? classes.grid : null,
          )}
        >
          {Object.entries(metrics).map(([metricName, metricValue], index) => (
            <Metric
              key={`metric-${index}`}
              name={metricName}
              value={metricValue}
            />
          ))}
        </div>
      ) : null}
    </>
  );
}
