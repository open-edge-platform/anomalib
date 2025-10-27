// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { omit } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { SourceConfig } from '../util';

export const useSourceMutation = (isNewSource: boolean) => {
    const addSource = $api.useMutation('post', '/api/sources', {
        meta: {
            invalidates: [['get', '/api/sources']],
        },
    });
    const updateSource = $api.useMutation('patch', '/api/sources/{source_id}', {
        meta: {
            invalidates: [['get', '/api/sources']],
        },
    });

    return async (body: SourceConfig) => {
        if (isNewSource) {
            const sourcePayload = {
                ...body,
                id: uuid(),
            };

            const response = await addSource.mutateAsync({ body: sourcePayload });

            return String(response.id);
        }

        const response = await updateSource.mutateAsync({
            params: { path: { source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        return String(response.id);
    };
};
