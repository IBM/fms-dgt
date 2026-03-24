// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';

import {
  DGT_TASK_CARD,
  DGT_TASK_RESULT,
  RunInformationCard,
} from '@/types/custom';
import { castDurationToString } from '@/src/common/utilities/time';

// ===================================================================================
//                               JSONL PARSING
// ===================================================================================
export function parseJsonl<T>(content: string): T[] {
  return JSON.parse(
    '[' +
      content
        .split(/\n/)
        .filter((l) => l.trim())
        .join(',') +
      ']',
  );
}

// ===================================================================================
//                               RUN DISCOVERY
// ===================================================================================
export function walk(
  directory: string,
  name: string,
  root: string,
): RunInformationCard[] {
  let runs: RunInformationCard[] = [];

  fs.readdirSync(directory).forEach((filename) => {
    const filePath = `${directory}/${filename}`;
    const stat = fs.statSync(filePath);

    if (stat && stat.isDirectory()) {
      const filenames = fs.readdirSync(filePath);

      if (
        filenames.includes('task_card.jsonl') &&
        filenames.includes('task_results.jsonl')
      ) {
        let startTime: Date = fs.statSync(`${filePath}/task_card.jsonl`).mtime;
        let duration = '-';
        let status = 'running';

        try {
          const resultsData = parseJsonl<DGT_TASK_RESULT>(
            fs.readFileSync(`${filePath}/task_results.jsonl`, 'utf8'),
          );
          const last = resultsData[resultsData.length - 1];

          startTime = new Date(last.start_time * 1000);
          const endTime = last.end_time
            ? new Date(last.end_time * 1000)
            : new Date();
          const [days, hours, mins, secs] = castDurationToString(
            (endTime.getTime() - startTime.getTime()) / 1000,
          );
          duration = `${days ? days + ' days ' : ''}${
            hours ? hours + ' hours ' : ''
          }${mins ? mins + ' mins ' : ''}${secs} sec`;
          status = last.status;
        } catch {
          try {
            const [days, hours, mins, secs] = castDurationToString(
              fs.statSync(`${filePath}/task_results.jsonl`).mtime.getTime() -
                startTime.getTime(),
            );
            duration = `${days ? days + ' days ' : ''}${
              hours ? hours + ' hours ' : ''
            }${mins ? mins + ' mins ' : ''}${secs} sec`;
          } catch {}
        }

        runs.push({
          name: name ? `${name}/${filename}` : filename,
          path: `${root}${name}/${filename}`,
          status,
          startTime,
          duration,
        });
      } else {
        runs = runs.concat(walk(filePath, `${name}/${filename}`, root));
      }
    }
  });

  return runs;
}

// ===================================================================================
//                               RUN DETAIL LOADERS
// ===================================================================================
export function loadTaskCards(path: string): DGT_TASK_CARD[] {
  return parseJsonl<DGT_TASK_CARD>(
    fs.readFileSync(`${path}/task_card.jsonl`, 'utf8'),
  );
}

export function loadLog(path: string): string | undefined {
  // Prefer logs.jsonl (public-repo LogDatastoreHandler format).
  try {
    const entries = fs
      .readFileSync(`${path}/logs.jsonl`, 'utf8')
      .split('\n')
      .filter((l) => l.trim())
      .map((l) => {
        const e = JSON.parse(l);
        return `${e.timestamp} ${e.level.padEnd(8)} [${e.logger}:${e.lineno}] ${
          e.message
        }`;
      });
    if (entries.length > 0) return entries.join('\n');
  } catch {}

  // Fall back to plain-text .log files (older internal-repo format, sorted by mtime for resumed runs).
  try {
    const logsDir = `${path}/logs`;
    const logFiles = fs
      .readdirSync(logsDir)
      .filter((f) => f.endsWith('.log'))
      .sort(
        (a, b) =>
          fs.statSync(`${logsDir}/${a}`).mtimeMs -
          fs.statSync(`${logsDir}/${b}`).mtimeMs,
      );
    if (logFiles.length > 0) {
      return logFiles
        .map((f) => fs.readFileSync(`${logsDir}/${f}`, 'utf8'))
        .join('\n');
    }
  } catch {}

  return undefined;
}

export function loadResults(path: string): { [key: string]: DGT_TASK_RESULT } {
  const taskResults = parseJsonl<DGT_TASK_RESULT>(
    fs.readFileSync(`${path}/task_results.jsonl`, 'utf8'),
  );
  return Object.fromEntries(taskResults.map((r) => [String(r.PID), r]));
}

export function loadDataPoints(path: string): {
  intermediate: any[];
  postprocessed: { [key: string]: any[] };
  final: any[];
  formatted: any[];
} {
  const data = {
    intermediate: [],
    postprocessed: {},
    final: [],
    formatted: [],
  } as {
    intermediate: any[];
    postprocessed: { [key: string]: any[] };
    final: any[];
    formatted: any[];
  };

  const parseJsonlFile = (file: string) =>
    parseJsonl(fs.readFileSync(`${path}/${file}`, 'utf8'));

  fs.readdirSync(path).forEach((filename) => {
    try {
      if (filename === 'data.jsonl') {
        data.intermediate = parseJsonlFile(filename);
      } else if (filename.startsWith('postproc_data_')) {
        data.postprocessed[filename.slice(14, -6)] = parseJsonlFile(filename);
      } else if (filename === 'final_data.jsonl') {
        data.final = parseJsonlFile(filename);
      } else if (filename === 'formatted_data.jsonl') {
        data.formatted = parseJsonlFile(filename);
      }
    } catch {}
  });

  return data;
}

export function resolvePid(
  taskCard: DGT_TASK_CARD,
  results: { [key: string]: DGT_TASK_RESULT },
): string {
  if (taskCard.process_id != null) return String(taskCard.process_id);
  return String(
    Object.values(results)
      .map((r) => [r.PID, r.start_time] as [string | number, number])
      // @ts-ignore — toSorted not in all TS lib versions
      .toSorted(
        (a: [string | number, number], b: [string | number, number]) =>
          b[1] - a[1],
      )[0][0],
  );
}
