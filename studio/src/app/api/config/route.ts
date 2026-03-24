// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

export const dynamic = 'force-dynamic';

export async function GET() {
  return Response.json({
    outputDir: process.env.DGT_OUTPUT_DIR || 'output',
  });
}
