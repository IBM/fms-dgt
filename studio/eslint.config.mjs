// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

import nextConfig from 'eslint-config-next/core-web-vitals';
import prettierConfig from 'eslint-config-prettier';

export default [
  ...nextConfig,
  prettierConfig,
  {
    rules: {
      'react/no-unescaped-entities': 'off',
    },
  },
];
