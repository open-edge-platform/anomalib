import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { useWebRTCConnection, WebRTCConnectionState } from '../../../../components/stream/web-rtc-connection-provider';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { PipelineSwitch } from './pipeline-switch.component';

vi.mock('../../../../components/stream/web-rtc-connection-provider', () => ({
    useWebRTCConnection: vi.fn(),
}));

vi.mock('../../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: vi.fn(),
}));

describe('PipelineSwitch', () => {
    const renderApp = ({
        webRtcConfig = {},
        pipelineConfig = {},
        onSetSelectedMediaItem = vi.fn(),
    }: {
        webRtcConfig?: Partial<WebRTCConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
        onSetSelectedMediaItem?: () => void;
    } = {}) => {
        vi.mocked(useWebRTCConnection).mockReturnValue({
            status: 'idle',
            stop: vi.fn(),
            start: vi.fn(),
            webRTCConnectionRef: { current: null },
            ...webRtcConfig,
        });

        vi.mocked(useSelectedMediaItem).mockReturnValue({
            selectedMediaItem: undefined,
            onSetSelectedMediaItem,
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <MemoryRouter initialEntries={['/projects/123/inspect']}>
                    <Routes>
                        <Route path='/projects/:projectId/inspect' element={<PipelineSwitch />} />
                    </Routes>
                </MemoryRouter>
            </QueryClientProvider>
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('Switch rendering', () => {
        it('renders the switch with "Enabled" label', async () => {
            renderApp();

            expect(await screen.findByText('Enabled')).toBeVisible();
        });

        it('switch is selected when pipeline status is "running"', async () => {
            renderApp({ pipelineConfig: { status: 'running' } });

            expect(await screen.findByRole('switch')).toBeChecked();
        });

        it('switch is not selected when pipeline status is not "running"', async () => {
            renderApp({ pipelineConfig: { status: 'idle' } });

            expect(await screen.findByRole('switch')).not.toBeChecked();
        });
    });

    describe('Switch disabled states', () => {
        it('model is missing', async () => {
            renderApp({ pipelineConfig: { model: undefined } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('model id is empty', async () => {
            renderApp({ pipelineConfig: { model: { id: '' } as SchemaPipeline['model'] } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('pipeline status is not active', async () => {
            renderApp({ pipelineConfig: { status: 'idle' } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('WebRTC is connecting', async () => {
            renderApp({ webRtcConfig: { status: 'connecting' } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('source is missing', async () => {
            renderApp({ pipelineConfig: { source: undefined } });

            expect(await screen.findByRole('switch')).toBeEnabled();
        });

        it('sink is missing', async () => {
            renderApp({ pipelineConfig: { sink: undefined } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('both source and sink are missing', async () => {
            renderApp({ pipelineConfig: { source: undefined, sink: undefined } });

            expect(await screen.findByRole('switch')).toBeDisabled();
        });

        it('is enabled when pipeline is active and all required components are present', async () => {
            renderApp({ pipelineConfig: { status: 'active' } });

            expect(await screen.findByRole('switch')).toBeEnabled();
        });

        it('is enabled when pipeline status is "running"', async () => {
            renderApp({ pipelineConfig: { status: 'running' } });

            expect(await screen.findByRole('switch')).toBeEnabled();
        });
    });

    describe('Switch interactions', () => {
        it('calls runPipeline when switch is turned on', async () => {
            const mockStart = vi.fn();
            const runPipelineSpy = vi.fn();
            const mockOnSetSelectedMediaItem = vi.fn();

            server.use(
                http.post('/api/projects/{project_id}/pipeline:run', () => {
                    runPipelineSpy();
                    return HttpResponse.json({}, { status: 204 });
                })
            );

            renderApp({
                pipelineConfig: { status: 'active' },
                webRtcConfig: { start: mockStart },
                onSetSelectedMediaItem: mockOnSetSelectedMediaItem,
            });

            expect(await screen.findByRole('switch')).not.toBeChecked();

            await userEvent.click(await screen.findByRole('switch'));

            await waitFor(() => {
                expect(runPipelineSpy).toHaveBeenCalled();
                expect(mockStart).toHaveBeenCalled();
                expect(mockOnSetSelectedMediaItem).toHaveBeenCalledWith(undefined);
            });
        });

        it('stops inference when switch is turned off', async () => {
            const mockStart = vi.fn();
            const stopPipelineSpy = vi.fn();

            server.use(
                http.post('/api/projects/{project_id}/pipeline:stop', () => {
                    stopPipelineSpy();
                    return HttpResponse.json({}, { status: 204 });
                })
            );

            renderApp({ pipelineConfig: { status: 'running' }, webRtcConfig: { start: mockStart } });

            expect(await screen.findByRole('switch')).toBeChecked();

            await userEvent.click(await screen.findByRole('switch'));

            await waitFor(() => {
                expect(stopPipelineSpy).toHaveBeenCalled();
                expect(mockStart).not.toHaveBeenCalled();
            });
        });

        it('starts WebRTC connection after successful runPipeline', async () => {
            const mockStart = vi.fn();

            server.use(
                http.post('/api/projects/{project_id}/pipeline:run', () => HttpResponse.json({}, { status: 204 }))
            );

            renderApp({ pipelineConfig: { status: 'active' }, webRtcConfig: { start: mockStart } });

            await userEvent.click(await screen.findByRole('switch'));

            await waitFor(() => {
                expect(mockStart).toHaveBeenCalled();
            });
        });

        it('clears selected media item after successful runPipeline', async () => {
            const mockOnSetSelectedMediaItem = vi.fn();
            server.use(
                http.post('/api/projects/{project_id}/pipeline:run', () => HttpResponse.json({}, { status: 204 }))
            );

            renderApp({ pipelineConfig: { status: 'active' }, onSetSelectedMediaItem: mockOnSetSelectedMediaItem });

            await userEvent.click(await screen.findByRole('switch'));

            await waitFor(() => {
                expect(mockOnSetSelectedMediaItem).toHaveBeenCalledWith(undefined);
            });
        });
    });
});
