// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { isEmpty } from 'lodash';
import yaml from 'js-yaml';
import CodeMirror from '@uiw/react-codemirror';
import { langs } from '@uiw/codemirror-extensions-langs';

import { useState, useEffect, useRef, useCallback } from 'react';
import { EditorView } from '@codemirror/view';

import {
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  ProgressIndicator,
  ProgressStep,
  Button,
  Pagination,
} from '@carbon/react';
import {
  WarningAlt,
  Activity,
  Task,
  ModelBuilder,
  DataBlob,
  BusinessMetrics,
  Dashboard,
  StopFilled,
} from '@carbon/icons-react';

import {
  RunInformationCard,
  DGT_TASK_CARD,
  DataPoint,
  DGT_TASK_RESULT,
  GenerationStats,
  TransformationStats,
  TokenUsage,
} from '@/types/custom';
import { DATA_PAGE_SIZE } from '@/src/common/constants';
import { useTheme } from '@/src/common/state/theme';
import { useNotification } from '@/src/components/notification/Notification';
import DataPointsTable from '@/src/components/datapoints_table/DataPointsTable';
import RunMetrics from '@/src/components/run_metrics/RunMetrics';
import StatsTab from '@/src/components/run_metrics/StatsTab';

import classes from './Run.module.scss';

// ===================================================================================
//                               HELPER FUNCTIONS
// ===================================================================================
async function fetchRun(
  path: string,
  logOffset: number,
  page: number,
  pageSize: number,
  setTaskCard: Function,
  appendLog: (delta: string, newOffset: number) => void,
  setResult: Function,
  setDataPoints: Function,
  setIsRunning: (v: boolean) => void,
  setGenerationStats: Function,
  setTransformationStats: Function,
  createNotification: Function,
): Promise<void> {
  const params = new URLSearchParams({
    path,
    log_offset: String(logOffset),
    page: String(page),
    page_size: String(pageSize),
  });
  await fetch(`/api/data/run?${params}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    next: { revalidate: 0 },
  }).then(async (response) => {
    if (response.status !== 200) {
      createNotification({
        title: 'Failed to fetch run',
        subtitle: 'Please verify your location and try again.',
        kind: 'error',
        timeout: 5000,
      });
    } else {
      const data = await response.json();
      setTaskCard(data.card);
      appendLog(data.logDelta ?? '', data.logOffset ?? 0);
      setResult(data.result);
      setDataPoints(data.datapoints);
      setIsRunning(data.isRunning ?? false);
      setGenerationStats(data.generationStats ?? null);
      setTransformationStats(data.transformationStats ?? null);
    }
  });
}

async function fetchTokenUsage(path: string, setTokenUsage: Function) {
  try {
    const response = await fetch(`/api/telemetry/tokens?path=${path}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      next: { revalidate: 0 },
    });
    if (response.status === 200) {
      setTokenUsage(await response.json());
    }
  } catch {}
}

async function cancelRun(path: string, createNotification: Function) {
  await fetch(`/api/data/run?path=${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'cancel' }),
  }).then(async (response) => {
    if (response.status !== 200) {
      createNotification({
        title: 'Failed to cancel run',
        subtitle: 'Please try again later.',
        kind: 'error',
        timeout: 5000,
      });
    } else {
      createNotification({
        title: 'Run Cancelled',
        subtitle: 'Please refresh to see updated status.',
        kind: 'success',
        timeout: 3000,
      });
    }
  });
}

// ===================================================================================
//                               RENDER FUNCTIONS
// ===================================================================================
function RunHeader({
  run,
  onCancel,
}: {
  run: RunInformationCard;
  onCancel: Function;
}) {
  return (
    <div className={classes.runTile}>
      <div className={classes.runTileHeader}>
        <h3>{run.name}</h3>
        <div className={classes.runToolbar}>
          <Button
            kind="danger"
            renderIcon={StopFilled}
            iconDescription="Cancel"
            disabled={run.status !== 'running'}
            onClick={() => onCancel()}
          >
            Cancel
          </Button>
        </div>
      </div>
      <div className={classes.runTileBlock}>
        <div className={classes.runTileBlockElement}>
          <div className={classes.runTileBlockElementHeading}>Start Time</div>
          <div>{run.startTime.toLocaleString()}</div>
        </div>
        <div className={classes.runTileBlockElement}>
          <div className={classes.runTileBlockElementHeading}>Duration</div>
          <div>{run.duration}</div>
        </div>
        <div className={classes.runTileBlockElement}>
          <div className={classes.runTileBlockElementHeading}>Status</div>
          <div>{run.status}</div>
        </div>
      </div>
    </div>
  );
}

function DataBucket({
  records,
  total,
  isRunning,
  dataPage,
  onPageChange,
  emptyMessage,
}: {
  records: DataPoint[];
  total: number;
  isRunning: boolean;
  dataPage: number;
  onPageChange: (page: number) => void;
  emptyMessage: string;
}) {
  if (isEmpty(records)) {
    return (
      <div className={classes.tabPanelWarning}>
        <WarningAlt size={32} />
        <div className={classes.tabPanelWarningText}>{emptyMessage}</div>
      </div>
    );
  }
  return (
    <>
      {isRunning && (
        <p className={classes.liveDataNote}>
          Showing last {records.length} of {total} records. Pagination available
          after run completes.
        </p>
      )}
      <DataPointsTable datapoints={records} />
      {!isRunning && total > DATA_PAGE_SIZE && (
        <Pagination
          page={dataPage + 1}
          pageSize={DATA_PAGE_SIZE}
          totalItems={total}
          pageSizes={[DATA_PAGE_SIZE]}
          onChange={({ page: p }) => onPageChange(p - 1)}
        />
      )}
    </>
  );
}

function DataViewer({
  datapoints,
  isRunning,
  dataPage,
  onPageChange,
}: {
  datapoints: {
    intermediate: DataPoint[];
    intermediateTotal: number;
    postprocessed: { [key: string]: DataPoint[] };
    postprocessedTotal: { [key: string]: number };
    final: DataPoint[];
    finalTotal: number;
    formatted: DataPoint[];
    formattedTotal: number;
  };
  isRunning: boolean;
  dataPage: number;
  onPageChange: (page: number) => void;
}) {
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const postprocKey = Object.keys(datapoints.postprocessed)[0];

  return (
    <div className={classes.dataViewer}>
      <ProgressIndicator
        className={classes.dataSelector}
        currentIndex={currentIndex}
        spaceEqually={true}
        onChange={(index) => setCurrentIndex(index)}
      >
        <ProgressStep label="Intermediate" secondaryLabel="" />
        <ProgressStep
          label="Post-Processed"
          disabled={isEmpty(datapoints.postprocessed)}
        />
        <ProgressStep label="Final" secondaryLabel="" />
        <ProgressStep
          label="Formatted"
          disabled={isEmpty(datapoints.formatted)}
        />
      </ProgressIndicator>
      <div className={classes.stepContainer}>
        {currentIndex === 0 ? (
          <DataBucket
            records={datapoints.intermediate}
            total={datapoints.intermediateTotal}
            isRunning={isRunning}
            dataPage={dataPage}
            onPageChange={onPageChange}
            emptyMessage="Intermediate data is not yet available. Please try again soon."
          />
        ) : currentIndex === 1 ? (
          <DataBucket
            records={postprocKey ? datapoints.postprocessed[postprocKey] : []}
            total={
              postprocKey ? datapoints.postprocessedTotal[postprocKey] ?? 0 : 0
            }
            isRunning={isRunning}
            dataPage={dataPage}
            onPageChange={onPageChange}
            emptyMessage="Post-processed data is not yet available. Please try again soon."
          />
        ) : currentIndex === 2 ? (
          <DataBucket
            records={datapoints.final}
            total={datapoints.finalTotal}
            isRunning={isRunning}
            dataPage={dataPage}
            onPageChange={onPageChange}
            emptyMessage="Final data is not yet available. Please try again soon."
          />
        ) : (
          <DataBucket
            records={datapoints.formatted}
            total={datapoints.formattedTotal}
            isRunning={isRunning}
            dataPage={dataPage}
            onPageChange={onPageChange}
            emptyMessage="Formatted data is not yet available. Please try again soon."
          />
        )}
      </div>
    </div>
  );
}

// ===================================================================================
//                               MAIN FUNCTION
// ===================================================================================
export default function RunView({
  run,
  lastFetched,
}: {
  run: RunInformationCard;
  lastFetched?: number;
}) {
  const [selectedTabIndex, setSelectedTabIndex] = useState<number>(0);
  const [taskCard, setTaskCard] = useState<DGT_TASK_CARD>();
  const [logText, setLogText] = useState<string>('');
  const [logOffset, setLogOffset] = useState<number>(0);
  const [result, setResult] = useState<DGT_TASK_RESULT>();
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null);
  const [generationStats, setGenerationStats] =
    useState<GenerationStats | null>(null);
  const [transformationStats, setTransformationStats] =
    useState<TransformationStats | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(run.status === 'running');
  const [datapoints, setDataPoints] = useState<{
    intermediate: DataPoint[];
    intermediateTotal: number;
    postprocessed: { [key: string]: DataPoint[] };
    postprocessedTotal: { [key: string]: number };
    final: DataPoint[];
    finalTotal: number;
    formatted: DataPoint[];
    formattedTotal: number;
  }>({
    intermediate: [],
    intermediateTotal: 0,
    postprocessed: {},
    postprocessedTotal: {},
    final: [],
    finalTotal: 0,
    formatted: [],
    formattedTotal: 0,
  });
  const [dataPage, setDataPage] = useState<number>(0);

  const { createNotification } = useNotification();
  const theme = useTheme();
  const codeViewerTheme = theme['theme'] === 'g90' ? 'dark' : 'light';
  const logEditorRef = useRef<EditorView | null>(null);
  // Scroll to bottom whenever new log text arrives.
  useEffect(() => {
    const view = logEditorRef.current;
    if (!view) return;
    view.dispatch({
      effects: EditorView.scrollIntoView(view.state.doc.length, { y: 'end' }),
    });
  }, [logText]);
  // Ref so the fetch callback always reads the latest offset without being a
  // dependency of the fetch effect (which would re-fire on every log delta).
  const logOffsetRef = useRef<number>(0);
  useEffect(() => {
    logOffsetRef.current = logOffset;
  }, [logOffset]);

  const appendLog = useCallback((delta: string, newOffset: number) => {
    setLogOffset(newOffset);
    if (!delta) return;
    setLogText((prev) => (prev ? prev + '\n' + delta : delta));
  }, []);

  useEffect(() => {
    if (run.path) {
      fetchRun(
        run.path,
        logOffsetRef.current,
        dataPage,
        DATA_PAGE_SIZE,
        setTaskCard,
        appendLog,
        setResult,
        setDataPoints,
        setIsRunning,
        setGenerationStats,
        setTransformationStats,
        createNotification,
      );
      fetchTokenUsage(run.path, setTokenUsage);
    }
  }, [run.path, lastFetched, dataPage, appendLog, createNotification]);

  return (
    <div className={classes.page}>
      <RunHeader
        run={run}
        onCancel={() => cancelRun(run.path, createNotification)}
      />
      <div className={classes.tabs}>
        <Tabs
          onChange={(state) => setSelectedTabIndex(state.selectedIndex)}
          selectedIndex={selectedTabIndex}
        >
          <TabList contained fullWidth>
            <Tab renderIcon={Dashboard}>Stats</Tab>
            <Tab renderIcon={Activity}>Log</Tab>
            <Tab renderIcon={Task}>Task Config</Tab>
            <Tab renderIcon={ModelBuilder}>Builder Config</Tab>
            <Tab renderIcon={DataBlob}>Data</Tab>
            <Tab renderIcon={BusinessMetrics}>Metrics</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <StatsTab
                datapoints={datapoints}
                tokenUsage={tokenUsage}
                generationStats={generationStats}
                transformationStats={transformationStats}
                isRunning={run.status === 'running'}
              />
            </TabPanel>
            <TabPanel>
              <CodeMirror
                value={logText}
                extensions={[langs.textile()]}
                basicSetup={{
                  lineNumbers: true,
                  foldGutter: false,
                  highlightActiveLineGutter: false,
                  highlightActiveLine: false,
                }}
                editable={false}
                readOnly={true}
                theme={codeViewerTheme}
                onCreateEditor={(view) => {
                  logEditorRef.current = view;
                }}
              />
            </TabPanel>
            <TabPanel>
              {taskCard ? (
                <CodeMirror
                  value={yaml.dump(taskCard.task_spec.task_init)}
                  extensions={[langs.yaml()]}
                  basicSetup={{
                    lineNumbers: true,
                    foldGutter: false,
                    highlightActiveLineGutter: false,
                    highlightActiveLine: false,
                  }}
                  editable={false}
                  readOnly={true}
                  theme={codeViewerTheme}
                />
              ) : null}
            </TabPanel>
            <TabPanel>
              {taskCard ? (
                <CodeMirror
                  value={yaml.dump(taskCard.databuilder_spec)}
                  extensions={[langs.yaml()]}
                  basicSetup={{
                    lineNumbers: true,
                    foldGutter: false,
                    highlightActiveLineGutter: false,
                    highlightActiveLine: false,
                  }}
                  editable={false}
                  readOnly={true}
                  theme={codeViewerTheme}
                />
              ) : null}
            </TabPanel>
            <TabPanel>
              <DataViewer
                datapoints={datapoints}
                isRunning={isRunning}
                dataPage={dataPage}
                onPageChange={setDataPage}
              />
            </TabPanel>
            <TabPanel>
              {run.status === 'running' ? (
                <div className={classes.tabPanelWarning}>
                  <BusinessMetrics size={32} />
                  <div className={classes.tabPanelWarningText}>
                    Metrics are written at run completion. Check back when the
                    run finishes.
                  </div>
                </div>
              ) : (
                <RunMetrics metrics={result?.metrics} />
              )}
            </TabPanel>
          </TabPanels>
        </Tabs>
      </div>
    </div>
  );
}
