// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useRef } from 'react';

import { useStatusBar } from '../status-bar';

interface UploadProgress {
    completed: number;
    total: number;
    failed: number;
}

export const useUploadStatus = () => {
    const { setStatus, removeStatus } = useStatusBar();
    const abortControllerRef = useRef<AbortController | null>(null);

    const startUpload = useCallback(
        (total: number) => {
            abortControllerRef.current = new AbortController();

            setStatus({
                id: 'batch-upload',
                type: 'upload',
                message: 'Uploading images',
                detail: `0 / ${total}`,
                progress: 0,
                variant: 'info',
                isCancellable: false,
                onCancel: () => {
                    abortControllerRef.current?.abort();
                    removeStatus('batch-upload');
                },
            });
        },
        [setStatus, removeStatus]
    );

    const isAborted = useCallback(() => {
        return abortControllerRef.current?.signal.aborted ?? false;
    }, []);

    const updateProgress = useCallback(
        (progress: UploadProgress) => {
            const percent = Math.round((progress.completed / progress.total) * 100);
            const detail =
                progress.failed > 0
                    ? `${progress.completed} / ${progress.total} (${progress.failed} failed)`
                    : `${progress.completed} / ${progress.total}`;

            setStatus({
                id: 'batch-upload',
                type: 'upload',
                message: 'Uploading images',
                detail,
                progress: percent,
                variant: progress.failed > 0 ? 'warning' : 'info',
                isCancellable: false,
                onCancel: () => {
                    abortControllerRef.current?.abort();
                    removeStatus('batch-upload');
                },
            });
        },
        [setStatus, removeStatus]
    );

    const completeUpload = useCallback(
        (success: boolean, failed: number = 0) => {
            if (success && failed === 0) {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: 'Upload complete ✓',
                    variant: 'success',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            } else if (failed > 0) {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: `Upload complete (${failed} failed)`,
                    variant: 'warning',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 5000,
                });
            } else {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: 'Upload failed',
                    progress: 100,
                    variant: 'error',
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            }
        },
        [setStatus]
    );

    const cancelUpload = useCallback(() => {
        abortControllerRef.current?.abort();
        removeStatus('batch-upload');
    }, [removeStatus]);

    return { startUpload, updateProgress, completeUpload, cancelUpload, isAborted };
};
