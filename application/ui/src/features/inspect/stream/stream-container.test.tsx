import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { useWebRTCConnection, WebRTCConnectionState } from '../../../components/stream/web-rtc-connection-provider';
import { StreamContainer } from './stream-container';

vi.mock('../../../components/stream/web-rtc-connection-provider', () => ({
    useWebRTCConnection: vi.fn(),
}));

describe('StreamContainer', () => {
    const renderApp = ({
        webRtcConfig = {},
        pipelineConfig = {},
    }: {
        webRtcConfig?: Partial<WebRTCConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
    }) => {
        vi.mocked(useWebRTCConnection).mockReturnValue({
            start: vi.fn(),
            status: 'idle',
            stop: vi.fn(),
            webRTCConnectionRef: { current: null },
            ...webRtcConfig,
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
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
                http.post('/api/projects/{project_id}/pipeline:run', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json({}, { status: 204 });
                })
            );

            renderApp({ webRtcConfig: { status: 'idle', start: mockedStart }, pipelineConfig: { status: 'idle' } });

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

            renderApp({ webRtcConfig: { status: 'idle', start: mockedStart }, pipelineConfig: { status: 'running' } });

            const button = await screen.findByRole('button', { name: /Start stream/i });
            await userEvent.click(button);

            expect(mockedStart).toHaveBeenCalled();
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });
    });

    it('render loading state', async () => {
        renderApp({ webRtcConfig: { status: 'connecting' } });

        expect(await screen.findByLabelText('Loading...')).toBeVisible();
    });

    it('webRtc connected', async () => {
        renderApp({ webRtcConfig: { status: 'connected' } });

        expect(await screen.findByLabelText('stream player')).toBeVisible();
    });

    it('autoplay webRtc if pipeline is enabled', async () => {
        const mockedStart = vi.fn();
        renderApp({ webRtcConfig: { status: 'idle', start: mockedStart }, pipelineConfig: { status: 'running' } });

        expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        expect(mockedStart).toHaveBeenCalled();
    });
});
