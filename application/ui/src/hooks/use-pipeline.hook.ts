// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from 'src/api/client';

import { useProjectIdentifier } from './use-project-identifier.hook';

export const usePipeline = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });
};

const POLLING_INTERVAL = 5000;
export const usePipelineMetrics = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/metrics',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: (query) => (query.state.status === 'success' ? POLLING_INTERVAL : false),
            retry: false,
        }
    );
};

export const usePatchPipeline = (project_id: string) => {
    return $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });
};

export const useEnablePipeline = (project_id: string) => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });
};

export const useDisablePipeline = (project_id: string) => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });
};

export const useConnectSourceToPipeline = () => {
    const { projectId } = useProjectIdentifier();
    const pipeline = usePatchPipeline(projectId);

    return (source_id: string) =>
        pipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { source_id } });
};

export const useConnectSinkToPipeline = () => {
    const { projectId } = useProjectIdentifier();
    const pipeline = usePatchPipeline(projectId);

    return (sink_id: string) =>
        pipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { sink_id } });
};
