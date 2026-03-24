// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import React, { PropsWithChildren } from 'react';
import Split from 'react-split';

import classes from './SplitPane.module.scss';

interface Props extends PropsWithChildren {
  defaultSizes?: [number, number];
  minSize?: number;
}

export default function SplitPane({
  children,
  defaultSizes = [22, 78],
  minSize = 220,
}: Props) {
  return (
    <Split
      className={classes.split}
      sizes={defaultSizes}
      minSize={minSize}
      direction="horizontal"
      gutterSize={6}
      gutterAlign="center"
      snapOffset={0}
      gutter={() => {
        const gutter = document.createElement('div');
        gutter.className = classes.gutter;
        return gutter;
      }}
    >
      {children}
    </Split>
  );
}
