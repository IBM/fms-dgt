// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { createContext, useState, useContext } from 'react';
import { Theme } from '@carbon/react';

// ===================================================================================
//                               CONSTANTS
// ===================================================================================
const ThemeContext = createContext<{
  theme: 'g10' | 'g90' | undefined;
  set: (theme: 'g10' | 'g90' | undefined) => void;
}>({
  theme: 'g10',
  set(theme) {},
});

// ===================================================================================
//                               MAIN FUNCTIONS
// ===================================================================================
export function ThemeProvider({ children }: { children: any }) {
  const [theme, setTheme] = useState<'g10' | 'g90' | undefined>('g10');

  const set = (theme: 'g10' | 'g90' | undefined) => {
    setTheme(theme);
  };

  return (
    <ThemeContext.Provider value={{ theme, set }}>
      <Theme theme={theme}>{children}</Theme>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
