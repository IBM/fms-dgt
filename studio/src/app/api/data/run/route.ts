// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { NextRequest } from 'next/server';
import fs from 'node:fs';

import { DGT_TASK_RESULT } from '@/types/custom';
import { loadTaskCards, loadLog, loadResults, loadDataPoints, resolvePid } from '@/src/lib/run';

// forces the route handler to be dynamic
export const dynamic = 'force-dynamic';

function cancelRun(path: string) {
  const taskCards = loadTaskCards(path);
  const taskCard = taskCards[taskCards.length - 1];

  let pid: string | undefined;

  if (taskCard.process_id) {
    pid = String(taskCard.process_id);
  }

  if (!pid) {
    const results = loadResults(path);
    pid = String(
      Object.values(results)
        .filter((r) => r.status === 'running')
        .map((r) => [r.PID, r.start_time] as [string | number, number])
        // @ts-ignore
        .toSorted((a: [string | number, number], b: [string | number, number]) => b[1] - a[1])[0][0],
    );
  }

  process.kill(Number(pid), 'SIGINT');
}

export async function GET(request: NextRequest) {
  if (!request.nextUrl.searchParams?.get('path')) {
    return Response.json(null, { status: 400, statusText: 'Missing "path" from request.' });
  }

  const path = request.nextUrl.searchParams.get('path');

  if (!path || !fs.existsSync(path)) {
    return Response.json(null, { status: 400, statusText: 'Invalid "path" in request.' });
  }

  try {
    const taskCards = loadTaskCards(path);
    const log = loadLog(path);
    const results = loadResults(path);
    const dataPoints = loadDataPoints(path);
    const taskCard = taskCards[taskCards.length - 1];
    const pid = resolvePid(taskCard, results);

    return Response.json({ card: taskCard, log, result: results[pid], datapoints: dataPoints });
  } catch {
    return Response.json(null, { status: 500, statusText: 'Failed to read run data' });
  }
}

export async function PUT(request: NextRequest) {
  if (!request.nextUrl.searchParams?.get('path')) {
    return Response.json(null, { status: 400, statusText: 'Missing "path" from request.' });
  }

  const path = request.nextUrl.searchParams.get('path');

  if (!path || !fs.existsSync(path)) {
    return Response.json(null, { status: 400, statusText: 'Invalid "path" in request.' });
  }

  const { action } = await request.json();

  if (action === 'cancel') {
    try {
      cancelRun(path);
    } catch {
      return Response.json(null, { status: 500, statusText: 'Failed to terminate the run' });
    }
    return Response.json(null);
  }
}
