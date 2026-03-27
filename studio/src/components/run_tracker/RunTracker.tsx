// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import cx from 'classnames';
import { useMemo, useState } from 'react';

import { Search, Select, SelectItem } from '@carbon/react';

import { RunInformationCard } from '@/types/custom';
import { splitRunName, formatDuration } from '@/src/lib/run_tracker';
import classes from './RunTracker.module.scss';

// ===================================================================================
//                               TYPES
// ===================================================================================
interface Props {
  runs: RunInformationCard[];
  selectedRun?: RunInformationCard;
  setSelectedRun: Function;
  disabled: boolean;
}

// ===================================================================================
//                               RENDER FUNCTION
// ===================================================================================
function StatusDot({ status }: { status: string }) {
  const dotClass =
    status === 'running'
      ? classes.statusDotRunning
      : status === 'completed'
        ? classes.statusDotCompleted
        : classes.statusDotErrored; // errored, cancelled, or anything unexpected

  return <span className={cx(classes.statusDot, dotClass)} />;
}

function SectionLabel({ label }: { label: string }) {
  return <div className={classes.sectionLabel}>{label}</div>;
}

function RunTile({
  run,
  selected = false,
  onSelect,
}: {
  run: RunInformationCard;
  selected?: boolean;
  onSelect: Function;
}) {
  const { title, path } = splitRunName(run.name);

  return (
    <div
      onClick={() => onSelect(run)}
      className={cx(classes.runTile, selected ? classes.selected : null)}
    >
      <div className={classes.runTileHeader}>
        <StatusDot status={run.status} />
        <div className={classes.runTileName}>
          <span className={classes.runTileTitle}>{title}</span>
          {path && <span className={classes.runTilePath}>{path}</span>}
        </div>
      </div>
      <div className={classes.runTileBlock}>
        <div className={classes.runTileBlockElement}>
          <div className={classes.runTileBlockElementHeading}>Start Time</div>
          <div className={classes.runTileBlockElementContent}>
            {run.startTime.toLocaleString()}
          </div>
        </div>
        <div className={classes.runTileBlockElement}>
          <div className={classes.runTileBlockElementHeading}>Duration</div>
          <div className={classes.runTileBlockElementContent}>
            {formatDuration(run.duration)}
          </div>
        </div>
      </div>
    </div>
  );
}

// ===================================================================================
//                               MAIN FUNCTION
// ===================================================================================
export default function RunTracker({
  runs,
  selectedRun,
  setSelectedRun,
  disabled = false,
}: Props) {
  const [runOrder, setRunOrder] = useState<string>('latest');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const { activeRuns, completedRuns, erroredRuns } = useMemo(() => {
    const filtered = searchQuery
      ? runs.filter((r) =>
          r.name.toLowerCase().includes(searchQuery.toLowerCase()),
        )
      : runs;

    const sort = (list: RunInformationCard[]) => {
      if (runOrder === 'a-z')
        return list.toSorted((a, b) => a.name.localeCompare(b.name));
      if (runOrder === 'z-a')
        return list.toSorted((a, b) => b.name.localeCompare(a.name));
      if (runOrder === 'oldest')
        return list.toSorted(
          (a, b) => a.startTime.getTime() - b.startTime.getTime(),
        );
      return list.toSorted(
        (a, b) => b.startTime.getTime() - a.startTime.getTime(),
      );
    };

    return {
      activeRuns: sort(filtered.filter((r) => r.status === 'running')),
      completedRuns: sort(filtered.filter((r) => r.status === 'completed')),
      erroredRuns: sort(filtered.filter((r) => r.status === 'errored')),
    };
  }, [runs, runOrder, searchQuery]);

  return (
    <div className={classes.page}>
      <div className={classes.search}>
        <Search
          id="search__run-tracker"
          labelText="Search Runs"
          placeholder="Type name"
          disabled={disabled}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <Select
          id="select__run-tracker__order"
          labelText="Order"
          defaultValue={'latest'}
          onChange={(event) => setRunOrder(event.target.value)}
          disabled={disabled}
        >
          <SelectItem value="a-z" text="A-Z" />
          <SelectItem value="z-a" text="Z-A" />
          <SelectItem value="latest" text="Latest" />
          <SelectItem value="oldest" text="Oldest" />
        </Select>
      </div>
      <div className={classes.runs}>
        {activeRuns.length > 0 && (
          <>
            <SectionLabel label="Running" />
            {activeRuns.map((run, index) => (
              <RunTile
                key={`run-active-${index}`}
                run={run}
                selected={selectedRun?.name === run.name}
                onSelect={setSelectedRun}
              />
            ))}
          </>
        )}
        {erroredRuns.length > 0 && (
          <>
            <SectionLabel label="Errored" />
            {erroredRuns.map((run, index) => (
              <RunTile
                key={`run-errored-${index}`}
                run={run}
                selected={selectedRun?.name === run.name}
                onSelect={setSelectedRun}
              />
            ))}
          </>
        )}
        {completedRuns.length > 0 && (
          <>
            <SectionLabel label="Completed" />
            {completedRuns.map((run, index) => (
              <RunTile
                key={`run-completed-${index}`}
                run={run}
                selected={selectedRun?.name === run.name}
                onSelect={setSelectedRun}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
