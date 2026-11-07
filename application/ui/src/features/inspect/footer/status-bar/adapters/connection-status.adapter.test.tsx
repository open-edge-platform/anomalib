// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { renderHook } from '@testing-library/react';

import { useStreamConnection } from '../../../../../components/stream/stream-connection-provider';
import { StatusBarProvider, useStatusBar } from '../status-bar-context';
import { ConnectionStatusAdapter } from './connection-status.adapter';

vi.mock('../../../../../components/stream/stream-connection-provider', () => ({
    useStreamConnection: vi.fn(),
}));

const wrapper = ({ children }: { children: ReactNode }) => (
    <StatusBarProvider>
        <ConnectionStatusAdapter />
        {children}
    </StatusBarProvider>
);

describe('ConnectionStatusAdapter', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('maps connected status', () => {
        vi.mocked(useStreamConnection).mockReturnValue({
            status: 'connected',
            start: vi.fn(),
            stop: vi.fn(),
            streamUrl: null,
            setStatus: vi.fn(),
        });

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        expect(result.current.connection).toBe('connected');
    });

    it('maps idle to disconnected', () => {
        vi.mocked(useStreamConnection).mockReturnValue({
            status: 'idle',
            start: vi.fn(),
            stop: vi.fn(),
            streamUrl: null,
            setStatus: vi.fn(),
        });

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        expect(result.current.connection).toBe('disconnected');
    });

    it('maps failed status', () => {
        vi.mocked(useStreamConnection).mockReturnValue({
            status: 'failed',
            start: vi.fn(),
            stop: vi.fn(),
            streamUrl: null,
            setStatus: vi.fn(),
        });

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        expect(result.current.connection).toBe('failed');
    });
});
