// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { StatusBarProvider, useStatusBar } from './status-bar-context';
import { StatusBar } from './status-bar.component';
import { ConnectionStatus, MainStatusState } from './status-bar.interface';

const StatusSetter = ({ status, connection }: { status?: MainStatusState; connection?: ConnectionStatus }) => {
    const { setStatus, removeStatus, setConnection } = useStatusBar();

    useEffect(() => {
        if (connection) {
            setConnection(connection);
        }

        if (status) {
            setStatus(status);
        } else {
            removeStatus('test');
        }
    }, [status, connection, setStatus, removeStatus, setConnection]);

    return null;
};

const renderStatusBar = (status?: MainStatusState, connection?: ConnectionStatus) => {
    return render(
        <StatusBarProvider>
            <StatusSetter status={status} connection={connection} />
            <StatusBar />
        </StatusBarProvider>
    );
};

describe('StatusBar', () => {
    describe('connection status labels', () => {
        it.each([
            ['connected', 'Connected'],
            ['connecting', 'Connecting...'],
            ['disconnected', 'Disconnected'],
            ['failed', 'Connection failed'],
        ] as const)('maps "%s" to "%s"', async (status, label) => {
            renderStatusBar(undefined, status);
            expect(await screen.findByText(label)).toBeInTheDocument();
        });
    });

    describe('cancel button', () => {
        it('shows cancel button only when isCancellable is true', async () => {
            const { rerender } = renderStatusBar({
                id: 'test',
                type: 'training',
                message: 'Training...',
                variant: 'info',
                isCancellable: false,
            });

            await screen.findByText('Training...');
            expect(screen.queryByText('Cancel')).not.toBeInTheDocument();

            rerender(
                <StatusBarProvider>
                    <StatusSetter
                        status={{
                            id: 'test',
                            type: 'training',
                            message: 'Training...',
                            variant: 'info',
                            isCancellable: true,
                            onCancel: vi.fn(),
                        }}
                    />
                    <StatusBar />
                </StatusBarProvider>
            );
            expect(await screen.findByText('Cancel')).toBeInTheDocument();
        });

        it('calls onCancel when clicked', async () => {
            const onCancel = vi.fn();
            renderStatusBar({
                id: 'test',
                type: 'training',
                message: 'Training...',
                variant: 'info',
                isCancellable: true,
                onCancel,
            });

            fireEvent.click(await screen.findByText('Cancel'));
            expect(onCancel).toHaveBeenCalledTimes(1);
        });
    });

    describe('progress bar', () => {
        it('shows indeterminate animation when progress is undefined', async () => {
            const { container } = renderStatusBar({
                id: 'test',
                type: 'export',
                message: 'Exporting...',
                variant: 'info',
            });

            await screen.findByText('Exporting...');
            const progressFill = container.querySelector('[class*="progressFill"]');
            expect(progressFill?.className).toContain('indeterminate');
        });

        it('shows determinate progress when progress is defined', async () => {
            const { container } = renderStatusBar({
                id: 'test',
                type: 'training',
                message: 'Training...',
                progress: 50,
                variant: 'info',
            });

            await screen.findByText('Training...');
            const progressFill = container.querySelector('[class*="progressFill"]');
            expect(progressFill?.className).not.toContain('indeterminate');
        });
    });
});
