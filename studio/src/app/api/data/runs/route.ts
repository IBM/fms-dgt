// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { NextRequest } from 'next/server';
import fs from 'node:fs';

import { walk } from '@/src/lib/run';

// forces the route handler to be dynamic
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  if (!request.nextUrl.searchParams?.get('path')) {
    return Response.json(null, { status: 400, statusText: 'Missing "path" from request.' });
  }

  const path = request.nextUrl.searchParams.get('path');

  if (!path || !fs.existsSync(path)) {
    return Response.json(null, { status: 400, statusText: 'Invalid "path" in request.' });
  }

  try {
    return Response.json(walk(path, '', path));
  } catch {
    return Response.json(null, { status: 500, statusText: 'Failed to fetch runs' });
  }
}
