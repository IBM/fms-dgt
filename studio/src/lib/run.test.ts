// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import {
  parseJsonl,
  loadLog,
  loadResults,
  loadDataPoints,
  loadTaskCards,
  resolvePid,
  walk,
} from './run';

// ===================================================================================
//                               FIXTURES
// ===================================================================================
let tmpDir: string;

function mkdir(...parts: string[]) {
  const p = path.join(tmpDir, ...parts);
  fs.mkdirSync(p, { recursive: true });
  return p;
}

function write(filePath: string, content: string) {
  fs.writeFileSync(path.join(tmpDir, filePath), content, 'utf8');
}

function writeJsonl(filePath: string, records: object[]) {
  write(filePath, records.map((r) => JSON.stringify(r)).join('\n'));
}

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'digit-studio-test-'));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

// ===================================================================================
//                               parseJsonl
// ===================================================================================
describe('parseJsonl', () => {
  it('parses a single-line JSONL string', () => {
    expect(parseJsonl<{ a: number }>('{"a":1}')).toEqual([{ a: 1 }]);
  });

  it('parses multi-line JSONL', () => {
    expect(parseJsonl<{ n: number }>('{"n":1}\n{"n":2}')).toEqual([
      { n: 1 },
      { n: 2 },
    ]);
  });

  it('ignores blank lines', () => {
    expect(parseJsonl<{ n: number }>('{"n":1}\n\n{"n":2}\n')).toEqual([
      { n: 1 },
      { n: 2 },
    ]);
  });
});

// ===================================================================================
//                               loadLog
// ===================================================================================
describe('loadLog', () => {
  it('prefers logs.jsonl over .log files', () => {
    writeJsonl('logs.jsonl', [
      {
        timestamp: '2026-01-01T00:00:00Z',
        level: 'INFO',
        logger: 'fms_dgt.test',
        lineno: 1,
        message: 'from jsonl',
      },
    ]);
    mkdir('logs');
    write('logs/1234.log', 'from dotlog');

    const result = loadLog(tmpDir);
    expect(result).toContain('from jsonl');
    expect(result).not.toContain('from dotlog');
  });

  it('falls back to .log files when logs.jsonl is absent', () => {
    mkdir('logs');
    write('logs/1234.log', 'log line one\n');

    expect(loadLog(tmpDir)).toBe('log line one\n');
  });

  it('concatenates multiple .log files sorted by mtime', () => {
    mkdir('logs');
    const first = path.join(tmpDir, 'logs', 'first.log');
    const second = path.join(tmpDir, 'logs', 'second.log');
    fs.writeFileSync(first, 'first\n');
    // ensure second has a strictly later mtime
    const later = new Date(Date.now() + 2000);
    fs.writeFileSync(second, 'second\n');
    fs.utimesSync(second, later, later);

    expect(loadLog(tmpDir)).toBe('first\n\nsecond\n');
  });

  it('returns undefined when neither source exists', () => {
    expect(loadLog(tmpDir)).toBeUndefined();
  });

  it('returns undefined when logs.jsonl is empty', () => {
    write('logs.jsonl', '');
    expect(loadLog(tmpDir)).toBeUndefined();
  });

  it('formats logs.jsonl entries correctly', () => {
    writeJsonl('logs.jsonl', [
      {
        timestamp: '2026-01-01T12:00:00Z',
        level: 'INFO',
        logger: 'fms_dgt.builder',
        lineno: 42,
        message: 'hello',
      },
    ]);
    const result = loadLog(tmpDir);
    expect(result).toBe(
      '2026-01-01T12:00:00Z INFO     [fms_dgt.builder:42] hello',
    );
  });
});

// ===================================================================================
//                               loadResults
// ===================================================================================
describe('loadResults', () => {
  it('keys results by stringified PID', () => {
    writeJsonl('task_results.jsonl', [
      {
        PID: 12345,
        status: 'completed',
        start_time: 1000,
        end_time: 2000,
        metrics: {},
      },
    ]);
    const results = loadResults(tmpDir);
    expect(results['12345']).toBeDefined();
    expect(results['12345'].status).toBe('completed');
  });

  it('handles numeric PID (not string) from JSON', () => {
    write(
      'task_results.jsonl',
      '{"PID": 99999, "status": "running", "start_time": 1000, "end_time": null, "metrics": {}}',
    );
    const results = loadResults(tmpDir);
    expect(results['99999']).toBeDefined();
  });

  it('last entry wins when same PID appears twice (resume case)', () => {
    writeJsonl('task_results.jsonl', [
      {
        PID: 42,
        status: 'running',
        start_time: 1000,
        end_time: null,
        metrics: {},
      },
      {
        PID: 42,
        status: 'completed',
        start_time: 1000,
        end_time: 2000,
        metrics: { 'Number of data produced': 5 },
      },
    ]);
    const results = loadResults(tmpDir);
    expect(results['42'].status).toBe('completed');
  });
});

// ===================================================================================
//                               loadDataPoints
// ===================================================================================
describe('loadDataPoints', () => {
  it('loads intermediate data from data.jsonl', () => {
    writeJsonl('data.jsonl', [{ question: 'q1', answer: 'a1' }]);
    const dp = loadDataPoints(tmpDir);
    expect(dp.intermediate).toHaveLength(1);
    expect(dp.intermediate[0].question).toBe('q1');
  });

  it('loads final data from final_data.jsonl', () => {
    writeJsonl('final_data.jsonl', [{ question: 'q1', answer: 'a1' }]);
    const dp = loadDataPoints(tmpDir);
    expect(dp.final).toHaveLength(1);
  });

  it('loads postprocessed data keyed by index', () => {
    writeJsonl('postproc_data_1.jsonl', [{ x: 1 }]);
    writeJsonl('postproc_data_2.jsonl', [{ x: 2 }, { x: 3 }]);
    const dp = loadDataPoints(tmpDir);
    expect(dp.postprocessed['1']).toHaveLength(1);
    expect(dp.postprocessed['2']).toHaveLength(2);
  });

  it('returns empty arrays when no data files exist', () => {
    const dp = loadDataPoints(tmpDir);
    expect(dp.intermediate).toEqual([]);
    expect(dp.final).toEqual([]);
    expect(dp.formatted).toEqual([]);
    expect(dp.postprocessed).toEqual({});
  });

  it('ignores unrelated files', () => {
    write('task_card.jsonl', '{}');
    write('dataloader_state.jsonl', '{}');
    const dp = loadDataPoints(tmpDir);
    expect(dp.intermediate).toEqual([]);
  });
});

// ===================================================================================
//                               resolvePid
// ===================================================================================
describe('resolvePid', () => {
  it('uses process_id from task card when present', () => {
    const card: any = { process_id: 55555 };
    const results = {};
    expect(resolvePid(card, results)).toBe('55555');
  });

  it('falls back to most recent result PID when process_id is absent', () => {
    const card: any = { process_id: undefined };
    const results: any = {
      '111': {
        PID: '111',
        status: 'completed',
        start_time: 1000,
        end_time: 2000,
        metrics: {},
      },
      '222': {
        PID: '222',
        status: 'completed',
        start_time: 2000,
        end_time: 3000,
        metrics: {},
      },
    };
    expect(resolvePid(card, results)).toBe('222');
  });
});

// ===================================================================================
//                               walk
// ===================================================================================
describe('walk', () => {
  function makeRun(dir: string, status: 'running' | 'completed') {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, 'task_card.jsonl'),
      JSON.stringify({ task_name: 'test', process_id: 1 }),
    );
    fs.writeFileSync(
      path.join(dir, 'task_results.jsonl'),
      JSON.stringify({
        PID: 1,
        status,
        start_time: 1741000000,
        end_time: status === 'completed' ? 1741000060 : null,
        metrics: {},
      }),
    );
  }

  it('discovers a run directory', () => {
    makeRun(path.join(tmpDir, 'myrun'), 'completed');
    const runs = walk(tmpDir, '', tmpDir);
    expect(runs).toHaveLength(1);
    expect(runs[0].name).toBe('myrun');
    expect(runs[0].status).toBe('completed');
  });

  it('discovers nested run directories', () => {
    makeRun(path.join(tmpDir, 'a', 'b', 'myrun'), 'completed');
    const runs = walk(tmpDir, '', tmpDir);
    expect(runs).toHaveLength(1);
    expect(runs[0].name).toBe('/a/b/myrun');
  });

  it('discovers multiple runs', () => {
    makeRun(path.join(tmpDir, 'run1'), 'completed');
    makeRun(path.join(tmpDir, 'run2'), 'running');
    const runs = walk(tmpDir, '', tmpDir);
    expect(runs).toHaveLength(2);
  });

  it('sets status from task_results.jsonl', () => {
    makeRun(path.join(tmpDir, 'active'), 'running');
    const runs = walk(tmpDir, '', tmpDir);
    expect(runs[0].status).toBe('running');
  });

  it('returns empty array for empty directory', () => {
    expect(walk(tmpDir, '', tmpDir)).toEqual([]);
  });
});
