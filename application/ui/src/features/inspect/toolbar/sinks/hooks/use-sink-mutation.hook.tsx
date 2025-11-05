import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { omit } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { SinkConfig } from '../utils';

export const useSinkMutation = (isNewSink: boolean) => {
    const { projectId } = useProjectIdentifier();
    const addSink = $api.useMutation('post', '/api/projects/{project_id}/sinks', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }]],
        },
    });
    const updateSink = $api.useMutation('patch', '/api/projects/{project_id}/sinks/{sink_id}', {
        params: { path: { project_id: projectId } },
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }]],
        },
    });

    return async (body: SinkConfig) => {
        if (isNewSink) {
            const id = uuid();
            const sinkPayload = { ...body, id };

            await addSink.mutateAsync({
                body: sinkPayload,
                params: { path: { project_id: projectId } },
            });

            return id;
        }

        const response = await updateSink.mutateAsync({
            params: { path: { project_id: projectId, sink_id: String(body.id) } },
            body: omit(body, 'sink_type'),
        });

        return String(response.id);
    };
};
