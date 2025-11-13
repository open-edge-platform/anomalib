// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useQueries } from '@tanstack/react-query';
import { $api, fetchClient } from 'src/api/client';

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

export const useEnablePipeline = ({ onSuccess }: { onSuccess?: () => void }) => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:run', {
        onSuccess,
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
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

export const useActivePipeline = () => {
    const projectsQuery = $api.useQuery('get', '/api/projects');
    const projectIds = projectsQuery.data?.projects?.map(({ id }) => String(id)) ?? [];

    const activePipelineResult = useQueries({
        queries: projectIds.map((projectId) => ({
            queryKey: ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            queryFn: async () => {
                const response = await fetchClient.GET('/api/projects/{project_id}/pipeline', {
                    params: { path: { project_id: projectId } },
                });
                return response.data;
            },
            enabled: projectIds.length > 0,
        })),
        combine: (results) => {
            return {
                data: results.find(({ data }) => data?.status === 'running')?.data,
                error: results.find(({ error }) => error)?.error,
                isLoading: results.some(({ isLoading }) => isLoading),
            };
        },
    });

    return {
        data: activePipelineResult.data,
        error: projectsQuery.error || activePipelineResult.error,
        isLoading: projectsQuery.isLoading || activePipelineResult.isLoading,
    };
};
