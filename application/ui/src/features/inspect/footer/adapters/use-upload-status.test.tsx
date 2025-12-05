// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { act, renderHook } from '@testing-library/react';

import { StatusBarProvider, useStatusBar } from '../status-bar';
import { useUploadStatus } from './use-upload-status';

const wrapper = ({ children }: { children: ReactNode }) => <StatusBarProvider>{children}</StatusBarProvider>;

describe('useUploadStatus', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('startUpload initializes with 0% progress', () => {
        const { result } = renderHook(
            () => ({
                uploadStatus: useUploadStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.uploadStatus.startUpload(10);
        });

        const status = result.current.statusBar.activeStatus;
        expect(status).not.toBeNull();
        expect(status?.message).toBe('Uploading images');
        expect(status?.detail).toBe('0 / 10');
        expect(status?.progress).toBe(0);
        expect(status?.variant).toBe('info');
    });

    it('updateProgress updates percentage and detail', () => {
        const { result } = renderHook(
            () => ({
                uploadStatus: useUploadStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.uploadStatus.startUpload(10);
        });

        act(() => {
            result.current.uploadStatus.updateProgress({ completed: 5, total: 10, failed: 0 });
        });

        const status = result.current.statusBar.activeStatus;
        expect(status?.detail).toBe('5 / 10');
        expect(status?.progress).toBe(50);
    });

    it('updateProgress shows warning variant when failures > 0', () => {
        const { result } = renderHook(
            () => ({
                uploadStatus: useUploadStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.uploadStatus.startUpload(10);
        });

        act(() => {
            result.current.uploadStatus.updateProgress({ completed: 5, total: 10, failed: 2 });
        });

        const status = result.current.statusBar.activeStatus;
        expect(status?.variant).toBe('warning');
        expect(status?.detail).toBe('5 / 10 (2 failed)');
    });

    it('completeUpload success sets success variant', () => {
        const { result } = renderHook(
            () => ({
                uploadStatus: useUploadStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.uploadStatus.startUpload(10);
        });

        act(() => {
            result.current.uploadStatus.completeUpload(true, 0);
        });

        expect(result.current.statusBar.activeStatus?.variant).toBe('success');
        expect(result.current.statusBar.activeStatus?.message).toBe('Upload complete ✓');
    });

    it('isAborted returns correct state', () => {
        const { result } = renderHook(
            () => ({
                uploadStatus: useUploadStatus(),
                statusBar: useStatusBar(),
            }),
            { wrapper }
        );

        act(() => {
            result.current.uploadStatus.startUpload(10);
        });

        expect(result.current.uploadStatus.isAborted()).toBe(false);

        act(() => {
            result.current.uploadStatus.cancelUpload();
        });

        expect(result.current.uploadStatus.isAborted()).toBe(true);
    });
});
