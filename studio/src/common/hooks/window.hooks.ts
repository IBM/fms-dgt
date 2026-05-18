// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from 'react';

export function useWindowResize() {
  const [WindowWidth, setWindowWidth] = useState<number>(
    global?.window && window.innerWidth,
  );
  const [WindowHeight, setWindowHeight] = useState<number>(
    global?.window && window.innerHeight,
  );

  useEffect(() => {
    const handleWindowResize = () => {
      setWindowWidth(window.innerWidth);
      setWindowHeight(window.innerHeight);
    };
    window.addEventListener('resize', handleWindowResize);
    return () => {
      window.removeEventListener('resize', handleWindowResize);
    };
  }, []);

  return { WindowWidth, WindowHeight };
}
