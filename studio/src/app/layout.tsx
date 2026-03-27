// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import HeaderView from '@/src/components/header/Header';
import { ThemeProvider } from '@/src/common/state/theme';
import { NotificationProvider } from '@/src/components/notification/Notification';

import '@/src/app/global.scss';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <NotificationProvider>
            <HeaderView />
            <main className="root">{children}</main>
          </NotificationProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
