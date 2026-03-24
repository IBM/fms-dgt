// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { truncate } from './string';

describe('truncate', () => {
  it('returns the string unchanged when under the limit', () => {
    expect(truncate('hello', 10)).toBe('hello');
  });

  it('returns the string unchanged when exactly at the limit', () => {
    expect(truncate('hello', 5)).toBe('hello');
  });

  it('truncates and appends ellipsis when over the limit', () => {
    expect(truncate('hello world', 5)).toBe('hello ...');
  });
});
