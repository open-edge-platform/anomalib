// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Text } from '@geti/ui';
import { clsx } from 'clsx';

import { ConnectionStatus } from './status-bar.interface';

import classes from './status-bar.module.scss';
import { useStatusBar } from './status-bar-context';

const CONNECTION_LABELS: Record<ConnectionStatus, string> = {
    connected: 'Connected',
    connecting: 'Connecting...',
    disconnected: 'Disconnected',
    failed: 'Connection failed',
};

export const StatusBar = () => {

    const { connection, activeStatus } = useStatusBar();
    const hasActiveStatus = activeStatus !== null;
    const isIndeterminate = activeStatus?.progress === undefined;

    return (
        <div className={classes.statusBar}>
            {/* Progress bar */}
            {activeStatus && (
                <div
                    className={clsx(
                        classes.progressFill,
                        classes[activeStatus.variant],
                        isIndeterminate && classes.indeterminate
                    )}
                    style={{ '--progress': `${activeStatus.progress ?? 100}%` }}
                />
            )}

            <div className={clsx(classes.contentWrapper, hasActiveStatus && classes.hasBackground)}>
                {/* WebRTC status */}
                <div className={classes.connectionSlot}>
                    <div
                        className={clsx(
                            classes.connectionDot,
                            hasActiveStatus ? classes.neutral : classes[connection]
                        )}
                    />
                    <Text UNSAFE_className={classes.connectionText}>{CONNECTION_LABELS[connection]}</Text>
                </div>

                {/* Main Status Area */}
                <div className={classes.mainStatusArea}>
                    {activeStatus ? (
                        <>
                            <Text UNSAFE_className={classes.message}>{activeStatus.message}</Text>
                            {activeStatus.detail && <Text UNSAFE_className={classes.detail}>{activeStatus.detail}</Text>}
                            {activeStatus.isCancellable && activeStatus.onCancel && (
                                <ActionButton isQuiet onPress={activeStatus.onCancel} UNSAFE_className={classes.cancelButton}>
                                    Cancel
                                </ActionButton>
                            )}
                        </>
                    ) : (
                        <Text UNSAFE_className={clsx(classes.message, classes.idleMessage)}>Idle</Text>
                    )}
                </div>
            </div>
        </div>
    );
};
