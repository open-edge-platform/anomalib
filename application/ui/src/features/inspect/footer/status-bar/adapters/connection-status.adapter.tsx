// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useStreamConnection } from '../../../../../components/stream/stream-connection-provider';
import { useStatusBar } from '../status-bar-context';
import type { ConnectionStatus } from '../status-bar.interface';

const CONNECTION_STATUS_MAP: Record<string, ConnectionStatus> = {
    connected: 'connected',
    connecting: 'connecting',
    failed: 'failed',
    idle: 'disconnected',
    disconnected: 'disconnected',
};

export const ConnectionStatusAdapter = () => {
    const { setConnection } = useStatusBar();
    const { status } = useStreamConnection();

    useEffect(() => {
        const connectionStatus = CONNECTION_STATUS_MAP[status] || 'disconnected';

        setConnection(connectionStatus);
    }, [status, setConnection]);

    return null;
};
