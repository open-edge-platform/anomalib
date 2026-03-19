import { useEffect } from 'react';

import { useSuspenseQuery } from '@tanstack/react-query';

import { getApiUrl } from '../api/client';

type StartupProjectSelectionSource = 'last_used' | 'active_pipeline' | 'first_project' | 'none';

export interface StartupProjectSelection {
    project_id: string | null;
    source: StartupProjectSelectionSource;
}

const PROJECT_STARTUP_SELECTION_QUERY_KEY = ['get', '/api/projects/startup-selection'] as const;

const fetchStartupProjectSelection = async (): Promise<StartupProjectSelection> => {
    const response = await fetch(getApiUrl('/api/projects/startup-selection'));
    if (!response.ok) {
        throw new Error(`Failed to resolve startup project: ${response.status}`);
    }

    return response.json() as Promise<StartupProjectSelection>;
};

const persistLastUsedProject = async (projectId: string): Promise<void> => {
    const response = await fetch(getApiUrl('/api/projects/last-used'), {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ project_id: projectId }),
    });

    if (!response.ok) {
        throw new Error(`Failed to persist last used project: ${response.status}`);
    }
};

export const useStartupProjectSelection = () =>
    useSuspenseQuery({
        queryKey: PROJECT_STARTUP_SELECTION_QUERY_KEY,
        queryFn: fetchStartupProjectSelection,
    });

export const usePersistLastUsedProject = (projectId: string) => {
    useEffect(() => {
        void persistLastUsedProject(projectId).catch(() => undefined);
    }, [projectId]);
};
