// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { splitRunName, formatDuration } from './run_tracker';

// ===================================================================================
//                               splitRunName
// ===================================================================================
describe('splitRunName', () => {
  it('splits a deep path into title and parent path', () => {
    expect(splitRunName('/public/examples/qa_ratings')).toEqual({
      title: 'qa_ratings',
      path: 'public/examples',
    });
  });

  it('handles a single-segment name with no parent', () => {
    expect(splitRunName('myrun')).toEqual({ title: 'myrun', path: '' });
  });

  it('strips leading slash', () => {
    expect(splitRunName('/myrun')).toEqual({ title: 'myrun', path: '' });
  });

  it('handles two-segment path', () => {
    expect(splitRunName('/core/simple')).toEqual({
      title: 'simple',
      path: 'core',
    });
  });
});

// ===================================================================================
//                               formatDuration
// ===================================================================================
describe('formatDuration', () => {
  it('compacts a full duration string', () => {
    expect(formatDuration('14 mins 17 sec')).toBe('14m 17s');
  });

  it('handles days, hours, mins, sec', () => {
    expect(formatDuration('1 days 2 hours 3 mins 4 sec')).toBe('1d 2h 3m 4s');
  });

  it('handles seconds only', () => {
    expect(formatDuration('11 sec')).toBe('11s');
  });

  it('handles singular forms', () => {
    expect(formatDuration('1 day 1 hour 1 min 1 sec')).toBe('1d 1h 1m 1s');
  });
});
