// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { omit } from 'lodash-es';

import { SinkConfig } from '../utils';

export const useSinkMutation = (isNewSink: boolean) => {
    const addSink = $api.useMutation('post', '/api/sinks', {
        meta: {
            invalidates: [['get', '/api/sinks']],
        },
    });
    const updateSink = $api.useMutation('patch', '/api/sinks/{sink_id}', {
        meta: {
            invalidates: [['get', '/api/sinks']],
        },
    });

    return async (body: SinkConfig) => {
        if (isNewSink) {
            const response = await addSink.mutateAsync({ body: omit(body, 'id') as SinkConfig });

            return String(response.id);
        }

        const response = await updateSink.mutateAsync({
            params: { path: { sink_id: String(body.id) } },
            body: omit(body, 'sink_type'),
        });

        return String(response.id);
    };
};
