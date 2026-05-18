// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { castDurationToString, calculateDuration } from './time';

describe('castDurationToString', () => {
  it('converts seconds only', () => {
    expect(castDurationToString(45)).toEqual([0, 0, 0, 45]);
  });

  it('converts minutes and seconds', () => {
    expect(castDurationToString(125)).toEqual([0, 0, 2, 5]);
  });

  it('converts hours, minutes, seconds', () => {
    expect(castDurationToString(3661)).toEqual([0, 1, 1, 1]);
  });

  it('converts days', () => {
    expect(castDurationToString(86400)).toEqual([1, 0, 0, 0]);
  });

  it('floors fractional seconds', () => {
    expect(castDurationToString(1.9)).toEqual([0, 0, 0, 1]);
  });
});

describe('calculateDuration', () => {
  it('returns undefineds when inputs are missing', () => {
    expect(calculateDuration(undefined, undefined)).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
  });

  it('calculates duration between two timestamps', () => {
    const start = 1000000000000;
    const end = start + 3661000; // 1h 1m 1s in ms
    expect(calculateDuration(end, start)).toEqual([0, 1, 1, 1]);
  });
});
