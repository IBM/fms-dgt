// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState, useEffect, useCallback } from 'react';
import { Loading } from '@carbon/react';

import { RunInformationCard } from '@/types/custom';
import RunTracker from '@/src/components/run_tracker/RunTracker';
import RunView from '@/src/components/run/Run';
import SplitPane from '@/src/components/split_pane/SplitPane';

import classes from './page.module.scss';

const POLL_INTERVAL_MS = 5000; // fast poll while a run is active
const IDLE_POLL_INTERVAL_MS = 30000; // slow poll when no runs are active

export default function Page() {
  const [outputDir, setOutputDir] = useState<string>('');
  const [runs, setRuns] = useState<RunInformationCard[]>([]);
  const [selectedRun, setSelectedRun] = useState<
    RunInformationCard | undefined
  >();
  const [loading, setLoading] = useState<boolean>(true);
  const [lastFetched, setLastFetched] = useState<number>(Date.now());

  // Fetch output dir from server at runtime (reads DGT_OUTPUT_DIR from Node process env)
  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => setOutputDir(data.outputDir));
  }, []);

  const fetchRuns = useCallback(
    async (dir: string) => {
      try {
        const response = await fetch(`/api/data/runs?path=${dir}`, {
          method: 'GET',
          cache: 'no-store',
        });
        if (response.ok) {
          const data: RunInformationCard[] = await response.json();
          const hydrated = data.map((r) => ({
            ...r,
            startTime: new Date(r.startTime),
          }));
          setRuns(hydrated);
          if (!selectedRun && hydrated.length > 0) {
            const active = hydrated.find((r) => r.status === 'running');
            const latest = hydrated.toSorted(
              (a, b) => b.startTime.getTime() - a.startTime.getTime(),
            )[0];
            setSelectedRun(active ?? latest);
          }
          if (selectedRun) {
            const freshStatus = hydrated.find(
              (r) => r.path === selectedRun.path,
            )?.status;
            // Refresh detail if selected run is still active, or just finished (transition to terminal state).
            if (
              freshStatus === 'running' ||
              (selectedRun.status === 'running' && freshStatus !== 'running')
            ) {
              setSelectedRun(
                (prev) => hydrated.find((r) => r.path === prev?.path) ?? prev,
              );
              setLastFetched(Date.now());
            }
          }
        }
      } finally {
        setLoading(false);
      }
    },
    [selectedRun],
  );

  // Load runs once outputDir is resolved
  useEffect(() => {
    if (outputDir) fetchRuns(outputDir);
  }, [outputDir]);

  // Fast poll while any run is active; slow background poll when idle to detect new runs.
  useEffect(() => {
    if (!outputDir) return;
    const hasActiveRun = runs.some((r) => r.status === 'running');
    const interval = setInterval(
      () => fetchRuns(outputDir),
      hasActiveRun ? POLL_INTERVAL_MS : IDLE_POLL_INTERVAL_MS,
    );
    return () => clearInterval(interval);
  }, [runs, outputDir, fetchRuns]);

  return (
    <div className={classes.layout}>
      <SplitPane>
        <div className={classes.sidebar}>
          {loading ? (
            <div className={classes.centered}>
              <Loading description="Loading runs..." withOverlay={false} />
            </div>
          ) : (
            <RunTracker
              runs={runs}
              selectedRun={selectedRun}
              setSelectedRun={setSelectedRun}
              disabled={loading}
            />
          )}
        </div>
        <div className={classes.detail}>
          {selectedRun ? (
            <RunView run={selectedRun} lastFetched={lastFetched} />
          ) : (
            <div className={classes.empty}>
              <p>Select a run to view details.</p>
            </div>
          )}
        </div>
      </SplitPane>
    </div>
  );
}
