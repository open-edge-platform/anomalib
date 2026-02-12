import { useEffect } from 'react';

import { useActivatePipeline, useActivePipeline } from '@anomalib-studio/hooks';
import isEmpty from 'lodash-es/isEmpty';

export const useEnsureActivePipeline = (projectId: string) => {
    const activePipelineMutation = useActivatePipeline({});
    const { data: activeProjectPipeline, isFetching } = useActivePipeline();

    const hasActiveProject = !isEmpty(activeProjectPipeline?.project_id);
    const isCurrentProjectActive = activeProjectPipeline?.project_id === projectId;

    useEffect(() => {
        if (!isFetching && !hasActiveProject && !activePipelineMutation.isPending) {
            activePipelineMutation.mutate({ params: { path: { project_id: projectId } } });
        }
    }, [activePipelineMutation, hasActiveProject, isFetching, projectId]);

    return { hasActiveProject, isCurrentProjectActive, activeProjectId: activeProjectPipeline?.project_id };
};
