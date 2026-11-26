// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { act, renderHook } from '@testing-library/react';

import { StatusBarProvider, useStatusBar } from './status-bar-context';
import { MainStatusState } from './status-bar.interface';

const wrapper = ({ children }: { children: ReactNode }) => <StatusBarProvider>{children}</StatusBarProvider>;

const createMockStatus = (overrides: Partial<MainStatusState> = {}): MainStatusState => ({
    id: 'test-status',
    type: 'training',
    message: 'Test message',
    variant: 'info',
    ...overrides,
});

describe('StatusBarContext', () => {
    describe('Connection Status', () => {
        it('initial state is disconnected', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });

            expect(result.current.connection).toBe('disconnected');
        });

        it('setConnection updates connection status', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });

            act(() => {
                result.current.setConnection('connected');
            });

            expect(result.current.connection).toBe('connected');
        });
    });

    describe('Status Management', () => {
        it('setStatus adds new status', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const status = createMockStatus({ id: 'training-1' });

            act(() => {
                result.current.setStatus(status);
            });

            expect(result.current.statuses.get('training-1')).toEqual(status);
        });

        it('setStatus updates existing status with same id', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const status1 = createMockStatus({ id: 'training-1', message: 'First message' });
            const status2 = createMockStatus({ id: 'training-1', message: 'Updated message' });

            act(() => {
                result.current.setStatus(status1);
            });

            act(() => {
                result.current.setStatus(status2);
            });

            expect(result.current.statuses.get('training-1')?.message).toBe('Updated message');
            expect(result.current.statuses.size).toBe(1);
        });

        it('removeStatus removes status from map', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const status = createMockStatus({ id: 'training-1' });

            act(() => {
                result.current.setStatus(status);
            });

            expect(result.current.statuses.has('training-1')).toBe(true);

            act(() => {
                result.current.removeStatus('training-1');
            });

            expect(result.current.statuses.has('training-1')).toBe(false);
        });
    });

    describe('Priority Selection (activeStatus)', () => {
        it('returns null when empty', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });

            expect(result.current.activeStatus).toBeNull();
        });

        it('returns highest priority status (training over export)', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const exportStatus = createMockStatus({ id: 'export-1', type: 'export', message: 'Exporting...' });
            const trainingStatus = createMockStatus({ id: 'training-1', type: 'training', message: 'Training...' });

            act(() => {
                result.current.setStatus(exportStatus);
                result.current.setStatus(trainingStatus);
            });

            expect(result.current.activeStatus?.type).toBe('training');
            expect(result.current.activeStatus?.message).toBe('Training...');
        });

        it('returns highest priority status (export over upload)', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const uploadStatus = createMockStatus({ id: 'upload-1', type: 'upload', message: 'Uploading...' });
            const exportStatus = createMockStatus({ id: 'export-1', type: 'export', message: 'Exporting...' });

            act(() => {
                result.current.setStatus(uploadStatus);
                result.current.setStatus(exportStatus);
            });

            expect(result.current.activeStatus?.type).toBe('export');
        });

        it('updates activeStatus when statuses change', () => {
            const { result } = renderHook(() => useStatusBar(), { wrapper });
            const trainingStatus = createMockStatus({ id: 'training-1', type: 'training' });
            const exportStatus = createMockStatus({ id: 'export-1', type: 'export' });

            act(() => {
                result.current.setStatus(trainingStatus);
            });

            expect(result.current.activeStatus?.type).toBe('training');

            act(() => {
                result.current.removeStatus('training-1');
                result.current.setStatus(exportStatus);
            });

            expect(result.current.activeStatus?.type).toBe('export');
        });
    });
});
