import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMetrics } from 'mocks/mock-metrics';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { StreamConnectionState, useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { StreamContainer } from './stream-container';

vi.mock('../../../components/stream/stream-connection-provider', () => ({
    useStreamConnection: vi.fn(),
}));

describe('StreamContainer', () => {
    const renderApp = ({
        streamConfig = {},
        pipelineConfig = {},
    }: {
        streamConfig?: Partial<StreamConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
    }) => {
        vi.mocked(useStreamConnection).mockReturnValue({
            stop: vi.fn(),
            start: vi.fn(),
            status: 'idle',
            streamUrl: null,
            setStatus: vi.fn(),
            ...streamConfig,
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            ),
            http.get('/api/projects/{project_id}/pipeline/metrics', ({ response }) =>
                response(200).json(getMockedMetrics({}))
            )
        );

        render(
            <QueryClientProvider client={new QueryClient()}>
                <ZoomProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect/stream']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect/stream' element={<StreamContainer />} />
                        </Routes>
                    </MemoryRouter>
                </ZoomProvider>
            </QueryClientProvider>
        );
    };

    describe('Start stream button', () => {
        it('call pipeline enable', async () => {
            const mockedStart = vi.fn();
            const pipelinePatchSpy = vi.fn();

            server.use(
                http.post('/api/projects/{project_id}/pipeline:activate', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json({}, { status: 204 });
                })
            );

            renderApp({
                streamConfig: { status: 'idle', start: mockedStart },
                pipelineConfig: { status: 'idle' },
            });

            const button = await screen.findByRole('button', { name: /Start stream/i });
            await userEvent.click(button);

            expect(mockedStart).toHaveBeenCalled();
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });

        it('pipeline enable is enabled', async () => {
            const mockedStart = vi.fn();
            const pipelinePatchSpy = vi.fn();

            server.use(
                http.post('/api/projects/{project_id}/pipeline:run', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json({}, { status: 204 });
                })
            );

            renderApp({ streamConfig: { status: 'idle', start: mockedStart }, pipelineConfig: { status: 'running' } });

            const button = await screen.findByRole('button', { name: /Start stream/i });
            await userEvent.click(button);

            expect(mockedStart).toHaveBeenCalled();
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });
    });

    it('renders stream while connecting', async () => {
        renderApp({ streamConfig: { status: 'connecting', streamUrl: '/api/stream' } });

        expect(await screen.findByLabelText('stream player')).toBeVisible();
    });

    it('renders stream when connected', async () => {
        renderApp({ streamConfig: { status: 'connected', streamUrl: '/api/stream' } });

        expect(await screen.findByLabelText('stream player')).toBeVisible();
    });

    it('autoplay stream if pipeline is enabled', async () => {
        const mockedStart = vi.fn();
        renderApp({ streamConfig: { status: 'idle', start: mockedStart }, pipelineConfig: { status: 'running' } });

        expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        expect(mockedStart).toHaveBeenCalled();
    });
});
