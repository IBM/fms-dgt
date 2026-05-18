// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function calculateDuration(
  endTimestamp?: number,
  startTimestamp?: number,
) {
  if (!endTimestamp || !startTimestamp) {
    return [undefined, undefined, undefined, undefined];
  }

  let duration = endTimestamp - startTimestamp;

  var durationInDays = Math.floor(duration / 1000 / 60 / 60 / 24);
  duration -= durationInDays * 1000 * 60 * 60 * 24;

  var durationInHours = Math.floor(duration / 1000 / 60 / 60);
  duration -= durationInHours * 1000 * 60 * 60;

  var durationInMinutes = Math.floor(duration / 1000 / 60);
  duration -= durationInMinutes * 1000 * 60;

  var durationInSeconds = Math.floor(duration / 1000);

  return [
    durationInDays,
    durationInHours,
    durationInMinutes,
    durationInSeconds,
  ];
}

/**
 * @param duration duration in seconds
 */
export function castDurationToString(duration: number) {
  var durationInDays = Math.floor(duration / 60 / 60 / 24);
  duration -= durationInDays * 60 * 60 * 24;

  var durationInHours = Math.floor(duration / 60 / 60);
  duration -= durationInHours * 60 * 60;

  var durationInMinutes = Math.floor(duration / 60);
  duration -= durationInMinutes * 60;

  var durationInSeconds = Math.floor(duration);

  return [
    durationInDays,
    durationInHours,
    durationInMinutes,
    durationInSeconds,
  ];
}
