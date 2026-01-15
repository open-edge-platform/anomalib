// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useStatusBar } from '../status-bar-context';

const getExportStatusId = (modelId: string) => `export-${modelId}`;

export const useExportStatus = () => {
    const { setStatus, statuses } = useStatusBar();

    const startExport = useCallback(
        (modelId: string, modelName: string, format: string) => {
            setStatus({
                id: getExportStatusId(modelId),
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
        (modelId: string, success: boolean) => {
            if (success) {
                setStatus({
                    id: getExportStatusId(modelId),
                    type: 'export',
                    message: 'Export complete ✓',
                    variant: 'success',
                    progress: 100,
                    isCancellable: false,
                    autoRemoveDelay: 3000,
                });
            } else {
                setStatus({
                    id: getExportStatusId(modelId),
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

    const isExporting = useCallback(
        (modelId: string) => {
            const status = statuses.get(getExportStatusId(modelId));
            return status?.variant === 'info';
        },
        [statuses]
    );

    return { startExport, completeExport, isExporting };
};
