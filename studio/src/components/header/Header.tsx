// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

import {
  Header,
  HeaderName,
  HeaderGlobalBar,
  HeaderGlobalAction,
} from '@carbon/react';
import { Awake, Asleep } from '@carbon/icons-react';

import { useTheme } from '@/src/common/state/theme';

import classes from './Header.module.scss';

// ===================================================================================
//                               MAIN FUNCTION
// ===================================================================================
export default function HeaderView() {
  const { theme, set } = useTheme();
  const [connected, setConnected] = useState<boolean>(true);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/health', { cache: 'no-store' });
        setConnected(res.ok);
      } catch {
        setConnected(false);
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Header aria-label="DiGiT Studio">
      <HeaderName as={Link} href="/" prefix="DiGiT">
        Studio
      </HeaderName>
      <HeaderGlobalBar>
        <span
          className={classes.statusDot}
          title={connected ? 'Connected' : 'Disconnected'}
          style={{ backgroundColor: connected ? '#24a148' : '#8d8d8d' }}
        />
        <HeaderGlobalAction
          aria-label={
            theme === 'g10' ? 'Switch to dark mode' : 'Switch to light mode'
          }
          onClick={() => {
            theme === 'g10' ? set('g90') : set('g10');
          }}
        >
          {theme === 'g10' ? <Asleep size={20} /> : <Awake size={20} />}
        </HeaderGlobalAction>
      </HeaderGlobalBar>
    </Header>
  );
}
