// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Loading } from '@carbon/react';

import { RunInformationCard } from '@/types/custom';
import { POLL_INTERVAL_MS } from '@/src/common/constants';
import RunTracker from '@/src/components/run_tracker/RunTracker';
import RunView from '@/src/components/run/Run';
import SplitPane from '@/src/components/split_pane/SplitPane';

import classes from './page.module.scss';

export default function Page() {
  const [outputDir, setOutputDir] = useState<string>('');
  const [runs, setRuns] = useState<RunInformationCard[]>([]);
  const [selectedRun, setSelectedRun] = useState<
    RunInformationCard | undefined
  >();
  const [loading, setLoading] = useState<boolean>(true);
  const [lastFetched, setLastFetched] = useState<number>(Date.now());

  // Stable ref so fetchRuns can read current selectedRun without being a dep.
  const selectedRunRef = useRef<RunInformationCard | undefined>(undefined);
  useEffect(() => {
    selectedRunRef.current = selectedRun;
  }, [selectedRun]);

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
          const current = selectedRunRef.current;
          if (!current && hydrated.length > 0) {
            const active = hydrated.find((r) => r.status === 'running');
            const latest = hydrated.toSorted(
              (a, b) => b.startTime.getTime() - a.startTime.getTime(),
            )[0];
            setSelectedRun(active ?? latest);
          }
          if (current) {
            const freshStatus = hydrated.find((r) => r.path === current.path)
              ?.status;
            // Refresh detail if selected run is still active, or just finished (transition to terminal state).
            if (
              freshStatus === 'running' ||
              (current.status === 'running' && freshStatus !== 'running')
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
    [], // stable — reads selectedRun via ref
  );

  // Load runs once outputDir is resolved
  useEffect(() => {
    if (outputDir) fetchRuns(outputDir);
  }, [outputDir, fetchRuns]);

  // Poll at fixed interval — fetchRuns is stable so this effect never re-runs.
  useEffect(() => {
    if (!outputDir) return;
    const interval = setInterval(() => fetchRuns(outputDir), POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [outputDir, fetchRuns]);

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
            <RunView
              key={selectedRun.path}
              run={selectedRun}
              lastFetched={lastFetched}
            />
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
