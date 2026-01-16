// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react';

import { useStatusBar } from '../status-bar-context';

interface UploadProgress {
    completed: number;
    total: number;
    failed: number;
}

const INITIAL_PROGRESS: UploadProgress = { completed: 0, total: 0, failed: 0 };

export const useUploadStatus = () => {
    const { setStatus, removeStatus } = useStatusBar();
    const abortControllerRef = useRef<AbortController | null>(null);
    const [progress, setProgress] = useState<UploadProgress>(INITIAL_PROGRESS);

    const processed = progress.completed + progress.failed;
    const isUploading = progress.total > 0 && processed < progress.total;
    const isCompleted = progress.total > 0 && processed === progress.total;

    useEffect(() => {
        if (!isUploading) return;

        const percent = Math.round((processed / progress.total) * 100);
        const detail =
            progress.failed > 0
                ? `${processed} / ${progress.total} (${progress.failed} failed)`
                : `${processed} / ${progress.total}`;

        setStatus({
            id: 'batch-upload',
            type: 'upload',
            message: 'Uploading images',
            detail,
            progress: percent,
            variant: progress.failed > 0 ? 'warning' : 'info',
            isCancellable: false,
        });
    }, [isUploading, processed, progress.total, progress.failed, setStatus]);

    useEffect(() => {
        if (!isCompleted) return;

        const { failed, completed, total } = progress;
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

        setProgress(INITIAL_PROGRESS);
    }, [isCompleted, progress, setStatus]);

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

    const incrementProgress = useCallback((success: boolean) => {
        setProgress((prev) =>
            success ? { ...prev, completed: prev.completed + 1 } : { ...prev, failed: prev.failed + 1 }
        );
    }, []);

    const cancelUpload = useCallback(() => {
        abortControllerRef.current?.abort();
        setProgress(INITIAL_PROGRESS);
        removeStatus('batch-upload');
    }, [removeStatus]);

    return { startUpload, incrementProgress, cancelUpload, isAborted, progress };
};
