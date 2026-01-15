// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { omit } from 'lodash-es';

import { SourceConfig } from '../util';

export const useSourceMutation = (isNewSource: boolean) => {
    const { projectId } = useProjectIdentifier();
    const addSource = $api.useMutation('post', '/api/projects/{project_id}/sources', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/sources', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
    const updateSource = $api.useMutation('patch', '/api/projects/{project_id}/sources/{source_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/sources', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    return async (body: SourceConfig) => {
        if (isNewSource) {
            // Omit id and project_id when creating - they're auto-generated/injected from URL
            const sourcePayload = omit(body, ['id', 'project_id']) as Parameters<
                typeof addSource.mutateAsync
            >[0]['body'];

            const response = await addSource.mutateAsync({
                body: sourcePayload,
                params: { path: { project_id: projectId } },
            });

            return String(response.id);
        }

        const response = await updateSource.mutateAsync({
            params: { path: { project_id: projectId, source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        return String(response.id);
    };
};
