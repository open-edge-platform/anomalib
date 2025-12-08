// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useRef, useState } from 'react';

import { useStatusBar } from '../status-bar-context';

interface UploadProgress {
    completed: number;
    total: number;
    failed: number;
}

export const useUploadStatus = () => {
    const { setStatus, removeStatus } = useStatusBar();
    const abortControllerRef = useRef<AbortController | null>(null);
    const [progress, setProgress] = useState<UploadProgress>({ completed: 0, total: 0, failed: 0 });

    const startUpload = useCallback(
        (total: number) => {
            abortControllerRef.current = new AbortController();
            setProgress({ completed: 0, total, failed: 0 });

            setStatus({
                id: 'batch-upload',
                type: 'upload',
                message: 'Uploading images',
                detail: `0 / ${total}`,
                progress: 0,
                variant: 'info',
                isCancellable: false,
            });
        },
        [setStatus]
    );

    const isAborted = useCallback(() => {
        return abortControllerRef.current?.signal.aborted ?? false;
    }, []);

    const updateStatusBar = useCallback(
        (newProgress: UploadProgress) => {
            const processed = newProgress.completed + newProgress.failed;
            const percent = Math.round((processed / newProgress.total) * 100);
            const detail =
                newProgress.failed > 0
                    ? `${processed} / ${newProgress.total} (${newProgress.failed} failed)`
                    : `${processed} / ${newProgress.total}`;

            setStatus({
                id: 'batch-upload',
                type: 'upload',
                message: 'Uploading images',
                detail,
                progress: percent,
                variant: newProgress.failed > 0 ? 'warning' : 'info',
                isCancellable: false,
            });
        },
        [setStatus]
    );

    const incrementProgress = useCallback(
        (success: boolean) => {
            setProgress((prev) => {
                const newProgress = success
                    ? { ...prev, completed: prev.completed + 1 }
                    : { ...prev, failed: prev.failed + 1 };
                updateStatusBar(newProgress);
                return newProgress;
            });
        },
        [updateStatusBar]
    );

    const completeUpload = useCallback(() => {
        setProgress((currentProgress) => {
            const { failed, completed, total } = currentProgress;
            const allFailed = completed === 0 && failed > 0;

            if (failed === 0) {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: 'Upload complete ✓',
                    variant: 'success',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            } else if (allFailed) {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: 'Upload failed',
                    progress: 100,
                    variant: 'error',
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            } else {
                setStatus({
                    id: 'batch-upload',
                    type: 'upload',
                    message: `Upload complete (${failed} of ${total} failed)`,
                    variant: 'warning',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 5000,
                });
            }

            return currentProgress;
        });
    }, [setStatus]);

    const cancelUpload = useCallback(() => {
        abortControllerRef.current?.abort();
        removeStatus('batch-upload');
    }, [removeStatus]);

    return { startUpload, incrementProgress, completeUpload, cancelUpload, isAborted, progress };
};
