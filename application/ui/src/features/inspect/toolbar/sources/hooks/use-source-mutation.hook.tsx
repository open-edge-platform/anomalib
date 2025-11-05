// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { omit } from 'lodash-es';
import { v4 as uuid } from 'uuid';

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
            const id = uuid();
            const sourcePayload = { ...body, id };

            await addSource.mutateAsync({
                body: sourcePayload,
                params: { path: { project_id: projectId } },
            });

            return id;
        }

        const response = await updateSource.mutateAsync({
            params: { path: { project_id: projectId, source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        return String(response.id);
    };
};
