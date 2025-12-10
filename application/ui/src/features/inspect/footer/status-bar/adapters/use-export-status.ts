// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useStatusBar } from '../status-bar-context';

export const useExportStatus = () => {
    const { setStatus } = useStatusBar();

    const startExport = useCallback(
        (modelName: string, format: string) => {
            setStatus({
                id: 'export',
                type: 'export',
                message: `Exporting ${modelName}...`,
                detail: `(${format})`,
                variant: 'info',
                isCancellable: false,
            });
        },
        [setStatus]
    );

    const completeExport = useCallback(
        (success: boolean) => {
            if (success) {
                setStatus({
                    id: 'export',
                    type: 'export',
                    message: 'Export complete ✓',
                    variant: 'success',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            } else {
                setStatus({
                    id: 'export',
                    type: 'export',
                    message: 'Export failed',
                    variant: 'error',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 5000,
                });
            }
        },
        [setStatus]
    );

    return { startExport, completeExport };
};
