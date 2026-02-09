import { toast } from '@geti/ui';
import { renderHook, waitFor } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { StreamConnectionStatus, useStreamConnection } from 'src/components/stream/stream-connection-provider';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';
import { queryClient } from 'src/query-client/query-client';

import { STREAM_ERROR_MESSAGE, useAutoPlayStream } from './use-auto-play-stream.hook';

vi.mock('../../../../components/stream/stream-connection-provider', async () => {
    const actual = await vi.importActual('../../../../components/stream/stream-connection-provider');
    return {
        ...actual,
        useStreamConnection: vi.fn(),
    };
});

vi.mock('@geti/ui', async () => {
    const actual = await vi.importActual('@geti/ui');
    return {
        ...actual,
        toast: vi.fn(),
    };
});

vi.mock('src/hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => ({ projectId: '123' }),
}));

describe('useAutoPlayStream', () => {
    const renderApp = ({
        status = 'idle',
        pipelineConfig,
    }: {
        status: StreamConnectionStatus;
        pipelineConfig?: Partial<SchemaPipeline> | null;
    }) => {
        const mockedStart = vi.fn();
        vi.mocked(useStreamConnection).mockReturnValue({
            stop: vi.fn(),
            start: mockedStart,
            status,
            streamUrl: null,
            setStatus: vi.fn(),
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline({ project_id: '123', ...pipelineConfig }))
            )
        );

        renderHook(() => useAutoPlayStream(), { wrapper: TestProviders });

        return mockedStart;
    };

    beforeEach(() => {
        queryClient.clear();
    });

    it('error message', async () => {
        const mockedStart = renderApp({ status: 'failed' });

        await waitFor(() => {
            expect(toast).toHaveBeenCalledWith({ type: 'error', message: STREAM_ERROR_MESSAGE });
            expect(mockedStart).not.toHaveBeenCalled();
        });
    });

    it('start stream', async () => {
        const mockedStart = renderApp({ status: 'idle' });

        await waitFor(() => {
            expect(mockedStart).toHaveBeenCalledWith();
        });
    });

    it('run pipeline', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'active';

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });
    });

    it('does not run pipeline if pipeline status is idle', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'idle';

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });

        await waitFor(() => {
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });
    });

    it('does not start stream if pipeline config is null', async () => {
        const mockedStartNull = renderApp({ status: 'idle', pipelineConfig: null });
        await waitFor(() => {
            expect(mockedStartNull).not.toHaveBeenCalled();
        });
    });

    it('does not start stream if status is connecting', async () => {
        const mockedStartConnecting = renderApp({ status: 'connecting' });
        await waitFor(() => {
            expect(mockedStartConnecting).not.toHaveBeenCalled();
        });
    });

    it('does not run pipeline if pipeline status is not idle or connected', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({ status: 'running' });

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });
        await waitFor(() => {
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });
    });

    it('shows error toast if pipeline run API fails', async () => {
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'active';
        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                return HttpResponse.json(
                    { detail: [{ msg: 'Failed', type: 'error', loc: ['pipeline'] }] },
                    { status: 500 }
                );
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });
        await waitFor(() => {
            expect(toast).toHaveBeenCalledWith({ type: 'error', message: STREAM_ERROR_MESSAGE });
        });
    });
});
