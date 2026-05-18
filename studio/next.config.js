// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

const isProd = process.env.NODE_ENV === 'production';

const nextConfig = {
  reactStrictMode: true,
  output: isProd ? 'export' : undefined,
  distDir: 'dist',
  sassOptions: {
    includePaths: [path.join(__dirname, 'styles')],
  },
  ...(!isProd && {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:4720/api/:path*',
        },
      ];
    },
  }),
};

module.exports = nextConfig;
