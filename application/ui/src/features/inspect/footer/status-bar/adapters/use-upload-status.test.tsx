// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { act, renderHook } from '@testing-library/react';

import { StatusBarProvider, useStatusBar } from '../status-bar-context';
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

    it('incrementProgress updates percentage and detail', () => {
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
            // Simulate 5 successful uploads
            for (let i = 0; i < 5; i++) {
                result.current.uploadStatus.incrementProgress(true);
            }
        });

        const status = result.current.statusBar.activeStatus;
        expect(status?.detail).toBe('5 / 10');
        expect(status?.progress).toBe(50);
    });

    it('incrementProgress shows warning variant when failures > 0', () => {
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
            // Simulate 3 successful and 2 failed uploads
            for (let i = 0; i < 3; i++) {
                result.current.uploadStatus.incrementProgress(true);
            }
            for (let i = 0; i < 2; i++) {
                result.current.uploadStatus.incrementProgress(false);
            }
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
            result.current.uploadStatus.startUpload(3);
        });

        act(() => {
            result.current.uploadStatus.incrementProgress(true);
            result.current.uploadStatus.incrementProgress(true);
            result.current.uploadStatus.incrementProgress(true);
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
