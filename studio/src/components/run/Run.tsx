// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { isEmpty } from 'lodash';
import yaml from 'js-yaml';
import CodeMirror from '@uiw/react-codemirror';
import { langs } from '@uiw/codemirror-extensions-langs';

import { useState, useEffect, useRef } from 'react';
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
} from '@carbon/react';
import {
  WarningAlt,
  Activity,
  Task,
  ModelBuilder,
  DataBlob,
  BusinessMetrics,
  StopFilled,
} from '@carbon/icons-react';

import {
  RunInformationCard,
  DGT_TASK_CARD,
  DataPoint,
  DGT_TASK_RESULT,
} from '@/types/custom';
import { useTheme } from '@/src/common/state/theme';
import { useNotification } from '@/src/components/notification/Notification';
import DataPointsTable from '@/src/components/datapoints_table/DataPointsTable';
import RunMetrics from '@/src/components/run_metrics/RunMetrics';

import classes from './Run.module.scss';

// ===================================================================================
//                               HELPER FUNCTIONS
// ===================================================================================
async function fetchRun(
  path: string,
  setTaskCard: Function,
  setLog: Function,
  setResult: Function,
  setDataPoints: Function,
  createNotification: Function,
) {
  await fetch(`/api/data/run?path=${path}`, {
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
      setLog(data.log);
      setResult(data.result);
      setDataPoints(data.datapoints);
    }
  });
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

function DataViewer({
  datapoints,
}: {
  datapoints: {
    intermediate: DataPoint[];
    postprocessed: { [key: string]: DataPoint[] };
    final: DataPoint[];
    formatted: DataPoint[];
  };
}) {
  const [currentIndex, setCurrentIndex] = useState<number>(0);

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
          !isEmpty(datapoints.intermediate) ? (
            <DataPointsTable datapoints={datapoints.intermediate} />
          ) : (
            <div className={classes.tabPanelWarning}>
              <WarningAlt size={32} />
              <div className={classes.tabPanelWarningText}>
                Intermediate data is not yet available. Please try again soon.
              </div>
            </div>
          )
        ) : currentIndex === 1 ? (
          <DataPointsTable
            datapoints={Object.values(datapoints.postprocessed)[0]}
          />
        ) : currentIndex === 2 ? (
          !isEmpty(datapoints.final) ? (
            <DataPointsTable datapoints={datapoints.final} />
          ) : (
            <div className={classes.tabPanelWarning}>
              <WarningAlt size={32} />
              <div className={classes.tabPanelWarningText}>
                Final data is not yet available. Please try again soon.
              </div>
            </div>
          )
        ) : !isEmpty(datapoints.formatted) ? (
          <DataPointsTable datapoints={datapoints.formatted} />
        ) : (
          <div className={classes.tabPanelWarning}>
            <WarningAlt size={32} />
            <div className={classes.tabPanelWarningText}>
              Formatted data is not yet available. Please try again soon.
            </div>
          </div>
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
  const [log, setLog] = useState<string>();
  const [result, setResult] = useState<DGT_TASK_RESULT>();
  const [datapoints, setDataPoints] = useState<{
    intermediate: DataPoint[];
    postprocessed: { [key: string]: DataPoint[] };
    final: DataPoint[];
    formatted: DataPoint[];
  }>({ intermediate: [], postprocessed: {}, final: [], formatted: [] });

  const { createNotification } = useNotification();
  const theme = useTheme();
  const codeViewerTheme = theme['theme'] === 'g90' ? 'dark' : 'light';
  const editorViewRef = useRef<EditorView | null>(null);

  useEffect(() => {
    if (run.path) {
      fetchRun(
        run.path,
        setTaskCard,
        setLog,
        setResult,
        setDataPoints,
        createNotification,
      );
    }
  }, [run.path, lastFetched]);

  // Auto-scroll log to bottom when new content arrives for a running job
  useEffect(() => {
    if (run.status === 'running' && editorViewRef.current) {
      const view = editorViewRef.current;
      view.dispatch({
        effects: EditorView.scrollIntoView(view.state.doc.length, { y: 'end' }),
      });
    }
  }, [log, run.status]);

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
            <Tab renderIcon={Activity}>Log</Tab>
            <Tab renderIcon={Task}>Task Config</Tab>
            <Tab renderIcon={ModelBuilder}>Builder Config</Tab>
            <Tab renderIcon={DataBlob}>Data</Tab>
            <Tab renderIcon={BusinessMetrics}>Metrics</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              {log || isEmpty(log) ? (
                <CodeMirror
                  value={log}
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
                    editorViewRef.current = view;
                  }}
                />
              ) : (
                <div className={classes.tabPanelWarning}>
                  <WarningAlt size={32} />
                  <div className={classes.tabPanelWarningText}>
                    Logs are unavailable at this moment
                  </div>
                </div>
              )}
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
              <DataViewer datapoints={datapoints} />
            </TabPanel>
            <TabPanel>
              <RunMetrics metrics={result?.metrics} />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </div>
    </div>
  );
}
