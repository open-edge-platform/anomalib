// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { $api } from '../api/client';

type StartupProjectSelectionSource = 'last_used' | 'active_pipeline' | 'first_project' | 'none';

export interface StartupProjectSelection {
    project_id: string | null;
    source: StartupProjectSelectionSource;
}

export const useStartupProjectSelection = () =>
    $api.useSuspenseQuery('get', '/api/projects/startup-selection');

export const usePersistLastUsedProject = (projectId: string) => {
    const mutation = $api.useMutation('put', '/api/projects/last-used');

    useEffect(() => {
        mutation.mutate(
            {
                body: {
                    project_id: projectId,
                },
            },
            {
                onError: (error) => {
                    console.error('Failed to persist last used project:', error);
                },
            },
        );
    }, [mutation, projectId]);
};
