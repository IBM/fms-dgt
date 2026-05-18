// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

export function splitRunName(name: string): { title: string; path: string } {
  const clean = name.replace(/^\//, '');
  const parts = clean.split('/');
  const title = parts[parts.length - 1];
  const path = parts.slice(0, -1).join('/');
  return { title, path };
}

export function formatDuration(duration: string): string {
  return duration
    .replace(' days', 'd')
    .replace(' day', 'd')
    .replace(' hours', 'h')
    .replace(' hour', 'h')
    .replace(' mins', 'm')
    .replace(' min', 'm')
    .replace(/ sec$/, 's')
    .replace(' sec ', 's ')
    .trim();
}
