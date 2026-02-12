// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fetchClient } from '@anomalib-studio/api';
import { useMutation } from '@tanstack/react-query';
import type { SchemaCompressionType, SchemaExportType } from 'src/api/openapi-spec';

import { useExportStatus } from '../../footer/status-bar/adapters/use-export-status';
import { downloadBlob, sanitizeFilename } from '../../utils';

interface ExportModelParams {
    projectId: string;
    projectName: string;
    modelId: string;
    modelName: string;
    format: SchemaExportType;
    formatLabel: string;
    compression: SchemaCompressionType | null;
}

export const useExportModel = () => {
    const { startExport, completeExport, isExporting } = useExportStatus();

    const mutation = useMutation({
        mutationFn: async (params: ExportModelParams) => {
            const { projectId, projectName, modelId, modelName, format, formatLabel, compression } = params;

            startExport(modelId, modelName, formatLabel);

            const response = await fetchClient.POST('/api/projects/{project_id}/models/{model_id}:export', {
                params: {
                    path: {
                        project_id: projectId,
                        model_id: modelId,
                    },
                },
                body: {
                    format,
                    compression,
                },
                parseAs: 'blob',
            });

            if (response.error) {
                throw new Error('Export failed');
            }

            const blob = response.data as Blob;
            const compressionSuffix = compression ? `_${compression}` : '';
            const sanitizedProjectName = sanitizeFilename(projectName);
            const sanitizedModelName = sanitizeFilename(modelName);
            const filename = `${sanitizedProjectName}_${sanitizedModelName}_${format}${compressionSuffix}.zip`;

            return { blob, filename, modelId };
        },
        onSuccess: ({ blob, filename, modelId }) => {
            completeExport(modelId, true);
            downloadBlob(blob, filename);
        },
        onError: (_error, variables) => {
            completeExport(variables.modelId, false);
        },
    });

    return { ...mutation, isExporting };
};
