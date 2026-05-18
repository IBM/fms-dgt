// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * Truncate string to specified character length
 */
export function truncate(text: string, length: number): string {
  if (text.length > length) {
    return text.slice(0, length) + ' ...';
  }
  return text;
}
