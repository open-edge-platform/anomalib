import { renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';
import { queryClient } from 'src/query-client/query-client';

import { useEnsureActivePipeline } from './use-ensure-active-pipeline.hook';

vi.mock('../../../../hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => ({ projectId: 'project-id-123' }),
}));

describe('useEnsureActivePipeline', () => {
    const mockProjectId = 'project-id-123';

    const renderHookWithProviders = (projectId: string) =>
        renderHook(() => useEnsureActivePipeline(projectId), { wrapper: TestProviders });

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    describe('Active Pipeline Detection', () => {
        it('returns hasActiveProject as true when there is an active pipeline', async () => {
            server.use(
                http.get('/api/active-pipeline', () => HttpResponse.json({ project_id: mockProjectId, status: 'idle' }))
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
                    HttpResponse.json({ project_id: activeProjectId, status: 'idle' })
                )
            );

            const { result } = renderHookWithProviders(currentProjectId);

            await waitFor(() => {
                expect(result.current.isCurrentProjectActive).toBe(false);
                expect(result.current.activeProjectId).toBe(activeProjectId);
            });
        });
    });

    describe('Auto-activation Behavior', () => {
        it('automatically activates pipeline when no active project exists', async () => {
            const activationSpy = vi.fn();

            server.use(
                http.get('/api/active-pipeline', () => HttpResponse.json(null)),
                http.post('/api/projects/{project_id}/pipeline:activate', () => {
                    activationSpy();
                    return HttpResponse.json({});
                })
            );

            renderHookWithProviders(mockProjectId);

            await waitFor(() => {
                expect(activationSpy).toHaveBeenCalled();
            });
        });

        it('does not activate pipeline when another project is already active', async () => {
            const activationSpy = vi.fn();

            server.use(
                http.get('/api/active-pipeline', () => HttpResponse.json({ project_id: '456', status: 'idle' })),
                http.post('/api/projects/{project_id}/pipeline:activate', () => {
                    activationSpy();
                    return HttpResponse.json({});
                })
            );

            renderHookWithProviders('123');

            await waitFor(() => expect(activationSpy).not.toHaveBeenCalled(), { timeout: 1000 });
        });

        it('does not activate pipeline when current project is already active', async () => {
            const activationSpy = vi.fn();

            server.use(
                http.get('/api/active-pipeline', () => {
                    return HttpResponse.json({ project_id: mockProjectId, status: 'idle' });
                }),
                http.post('/api/projects/{project_id}/pipeline:activate', () => {
                    activationSpy();
                    return HttpResponse.json({});
                })
            );

            renderHookWithProviders(mockProjectId);

            await waitFor(
                () => {
                    expect(activationSpy).not.toHaveBeenCalled();
                },
                { timeout: 1000 }
            );
        });
    });
});
