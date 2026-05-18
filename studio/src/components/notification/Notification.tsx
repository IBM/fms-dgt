// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { createContext, useContext, useState, useRef } from 'react';
import {
  ActionableNotification,
  InlineNotification,
  ToastNotification,
} from '@carbon/react';

import { Notification as NotificationType } from '@/types/custom';

import styles from './Notification.module.scss';

type NotificationContextType = {
  createNotification(notification: NotificationType): void;
};

const NotificationContext = createContext<NotificationContextType | undefined>(
  undefined,
);

const NotificationComponent = ({
  type,
  title,
  subtitle,
  kind,
  caption,
  onCloseButtonClick,
  onActionButtonClick,
}: NotificationType) =>
  type === 'Inline' ? (
    <InlineNotification
      hideCloseButton={false}
      className={styles.notification}
      title={title}
      kind={kind}
      subtitle={subtitle}
      onCloseButtonClick={onCloseButtonClick}
    />
  ) : type === 'Actionable' ? (
    <ActionableNotification
      hideCloseButton={false}
      className={styles.notification}
      title={title}
      kind={kind}
      subtitle={subtitle}
      onCloseButtonClick={onCloseButtonClick}
      onActionButtonClick={onActionButtonClick}
    />
  ) : (
    <ToastNotification
      hideCloseButton={true}
      className={styles.notification}
      title={title}
      kind={kind}
      subtitle={subtitle}
      caption={caption}
      onCloseButtonClick={onCloseButtonClick}
    />
  );

export const NotificationProvider = ({ children }: { children: any }) => {
  const [notification, setNotification] = useState<
    NotificationType | undefined
  >();
  const timeoutId = useRef<NodeJS.Timeout | undefined>(undefined);

  const createNotification = (notification: NotificationType) => {
    if (timeoutId.current) {
      clearTimeout(timeoutId.current);
    }
    setNotification(notification);
    timeoutId.current = setTimeout(
      () => setNotification(undefined),
      notification.timeout ? notification.timeout : 5000,
    );
  };

  return (
    <NotificationContext.Provider value={{ createNotification }}>
      {children}
      {notification ? <NotificationComponent {...notification} /> : null}
    </NotificationContext.Provider>
  );
};

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error(
      'useNotification must be used within a NotificationProvider',
    );
  }
  return context;
};
