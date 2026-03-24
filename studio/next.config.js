// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

const cspMap = {
  'base-uri': ["'none'"],
  'font-src': ["'self'", 'data:', "'unsafe-inline'"],
  'form-action': ["'self'"],
  'frame-src': ["'self'"],
  'img-src': ["'self'", 'data:', 'blob:'],
  'media-src': ["'self'", 'blob:'],
  'object-src': ["'none'"],
  'style-src': ["'self'", "'unsafe-inline'"],
};

const getCSPString = (cspMap) =>
  Object.entries(cspMap)
    .map(([key, values]) => `${key} ${values.join(' ')}`)
    .join('; ');

const headers = [
  { key: 'Content-Security-Policy', value: getCSPString(cspMap) },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-XSS-Protection', value: '1' },
  {
    key: 'Strict-Transport-Security',
    value: 'max-age=63072000; includeSubDomains; preload',
  },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
];

const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  sassOptions: {
    includePaths: [path.join(__dirname, 'styles')],
  },
  async headers() {
    return [
      { source: '/', headers },
      { source: '/:path*', headers },
    ];
  },
};

module.exports = nextConfig;
