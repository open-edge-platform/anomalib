import { useEffect } from 'react';

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { isNil } from 'lodash-es';

import { useGetProjects } from './use-get-project.hooks';

export const useSelectedProject = () => {
    const { projectId } = useProjectIdentifier();
    const { projects, isFetchingNextPage, hasNextPage, fetchNextPage } = useGetProjects();

    const selectedProject = projects.find((item) => item.id === projectId);

    useEffect(() => {
        if (isNil(projectId) || !isNil(selectedProject) || isFetchingNextPage || !hasNextPage) {
            return;
        }

        fetchNextPage();
    }, [fetchNextPage, hasNextPage, isFetchingNextPage, projectId, selectedProject]);

    return selectedProject;
};
