import { renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';
import { queryClient } from 'src/query-client/query-client';

import { useActivePipelineStatus } from './use-active-pipeline-status.hook';

vi.mock('../../../../hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => ({ projectId: 'project-id-123' }),
}));

describe('useActivePipelineStatus', () => {
    const mockProjectId = 'project-id-123';

    const renderHookWithProviders = (projectId: string) =>
        renderHook(() => useActivePipelineStatus(projectId), { wrapper: TestProviders });

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    describe('Active Pipeline Detection', () => {
        it('returns hasActiveProject as true when there is an active pipeline', async () => {
            server.use(
                http.get('/api/active-pipeline', () =>
                    HttpResponse.json({ project_id: mockProjectId, status: 'idle', inference_device: 'CPU' })
                )
            );

            const { result } = renderHookWithProviders('123');

            await waitFor(() => {
                expect(result.current?.hasActiveProject).toBe(true);
            });
        });

        it('returns hasActiveProject as false when there is no active pipeline', async () => {
            server.use(http.get('/api/active-pipeline', () => HttpResponse.json()));

            const { result } = renderHookWithProviders('123');

            await waitFor(() => {
                expect(result.current.hasActiveProject).toBe(false);
            });
        });

        it('returns correct activeProjectId when pipeline is active', async () => {
            const activeProjectId = '789';
            const currentProjectId = '321';
            server.use(
                http.get('/api/active-pipeline', () =>
                    HttpResponse.json({ project_id: activeProjectId, status: 'idle', inference_device: 'CPU' })
                )
            );

            const { result } = renderHookWithProviders(currentProjectId);

            await waitFor(() => {
                expect(result.current.isCurrentProjectActive).toBe(false);
                expect(result.current.activeProjectId).toBe(activeProjectId);
            });
        });
    });
});
