// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, type ReactNode } from 'react';

import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

import { server } from '../msw-node-setup';
import { TestProviders } from '../providers';
import { queryClient } from '../query-client/query-client';
import { usePersistLastUsedProject, useStartupProjectSelection } from './use-project-selection.hook';

const SuspenseProviders = ({ children }: { children: ReactNode }) => (
    <TestProviders>
        <Suspense fallback={null}>{children}</Suspense>
    </TestProviders>
);

describe('useProjectSelection', () => {
    beforeEach(() => {
        queryClient.clear();
    });

    it('reads the startup project selection from the backend', async () => {
        server.use(
            http.get('http://localhost:8000/api/projects/startup-selection', () =>
                HttpResponse.json({
                    project_id: 'project-2',
                    source: 'active_pipeline',
                })
            )
        );

        const { result } = renderHook(() => useStartupProjectSelection(), {
            wrapper: SuspenseProviders,
        });

        await waitFor(() => {
            expect(result.current.data).toEqual({
                project_id: 'project-2',
                source: 'active_pipeline',
            });
        });
    });

    it('stores the current project as the last used project', async () => {
        const persistSpy = vi.fn();

        server.use(
            http.put('http://localhost:8000/api/projects/last-used', async ({ request }) => {
                persistSpy(await request.json());
                return new HttpResponse(null, { status: 204 });
            })
        );

        renderHook(() => usePersistLastUsedProject('project-3'), {
            wrapper: TestProviders,
        });

        await waitFor(() => {
            expect(persistSpy).toHaveBeenCalledWith({ project_id: 'project-3' });
        });
    });
});
