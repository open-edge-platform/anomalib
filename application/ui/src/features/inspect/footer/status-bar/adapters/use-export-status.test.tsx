// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { act, renderHook } from '@testing-library/react';

import { StatusBarProvider, useStatusBar } from '../status-bar-context';
import { useExportStatus } from './use-export-status';

const wrapper = ({ children }: { children: ReactNode }) => <StatusBarProvider>{children}</StatusBarProvider>;

describe('useExportStatus', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('startExport sets status with model name and format', () => {
        const { result } = renderHook(
            () => ({
                exportStatus: useExportStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.exportStatus.startExport('123', 'MyModel', 'OpenVINO');
        });

        const status = result.current.statusBar.activeStatus;
        expect(status).not.toBeNull();
        expect(status?.message).toBe('Exporting MyModel...');
        expect(status?.detail).toBe('(OpenVINO)');
        expect(status?.variant).toBe('info');
    });

    it('completeExport success sets success variant and removes after timeout', () => {
        const { result } = renderHook(
            () => ({
                exportStatus: useExportStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.exportStatus.startExport('123', 'MyModel', 'OpenVINO');
        });

        act(() => {
            result.current.exportStatus.completeExport('123', true);
        });

        expect(result.current.statusBar.activeStatus?.variant).toBe('success');
        expect(result.current.statusBar.activeStatus?.message).toBe('Export complete ✓');

        act(() => {
            vi.advanceTimersByTime(3000);
        });

        expect(result.current.statusBar.activeStatus).toBeNull();
    });

    it('completeExport failure sets error variant and removes after timeout', () => {
        const { result } = renderHook(
            () => ({
                exportStatus: useExportStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.exportStatus.startExport('123', 'MyModel', 'OpenVINO');
        });

        act(() => {
            result.current.exportStatus.completeExport('123', false);
        });

        expect(result.current.statusBar.activeStatus?.variant).toBe('error');
        expect(result.current.statusBar.activeStatus?.message).toBe('Export failed');

        act(() => {
            vi.advanceTimersByTime(5000);
        });

        expect(result.current.statusBar.activeStatus).toBeNull();
    });
});
