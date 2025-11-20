import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { getMockedMediaItem } from '../../../../mocks/mock-media-item';
import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { useWebRTCConnection, WebRTCConnectionState } from '../../../components/stream/web-rtc-connection-provider';
import { MediaItem } from '../dataset/types';
import { InferenceProvider } from '../inference-provider.component';
import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { MainContent } from './main-content.component';
import { SOURCE_MESSAGE } from './source-sink-message/source-sink-message.component';

vi.mock('../../../components/stream/web-rtc-connection-provider', () => ({
    useWebRTCConnection: vi.fn(),
}));

vi.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: vi.fn(),
}));

describe('MainContent', () => {
    const mockMediaItem = getMockedMediaItem({});

    const renderApp = ({
        webRtcConfig = {},
        pipelineConfig = {},
        selectedMediaItem,
        activePipelineConfig = {},
    }: {
        webRtcConfig?: Partial<WebRTCConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
        selectedMediaItem?: MediaItem;
        activePipelineConfig?: Partial<SchemaPipeline> | null;
    }) => {
        vi.mocked(useWebRTCConnection).mockReturnValue({
            start: vi.fn(),
            status: 'idle',
            stop: vi.fn(),
            webRTCConnectionRef: { current: null },
            ...webRtcConfig,
        });

        vi.mocked(useSelectedMediaItem).mockReturnValue({ selectedMediaItem, onSetSelectedMediaItem: vi.fn() });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            ),
            http.get('/api/active-pipeline', ({ response }) => {
                if (activePipelineConfig === null) {
                    return response(200).json(null);
                }
                return response(200).json(getMockedPipeline({ project_id: '123', ...activePipelineConfig }));
            })
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ZoomProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route
                                path='/projects/:projectId/inspect'
                                element={
                                    <InferenceProvider>
                                        <MainContent />
                                    </InferenceProvider>
                                }
                            />
                        </Routes>
                    </MemoryRouter>
                </ZoomProvider>
            </QueryClientProvider>
        );
    };

    describe('SinkMessage', () => {
        it('renders when no source is configured and no media item selected', async () => {
            renderApp({ pipelineConfig: { source: undefined } });

            expect(await screen.findByText(SOURCE_MESSAGE)).toBeVisible();
        });

        it('does not render SinkMessage when no sink is configured', async () => {
            renderApp({ pipelineConfig: { sink: undefined } });

            await waitFor(() => {
                expect(screen.queryByText(SOURCE_MESSAGE)).not.toBeInTheDocument();
            });
        });

        it('renders when both source and sink are missing', async () => {
            renderApp({ pipelineConfig: { source: undefined, sink: undefined } });

            expect(await screen.findByText(SOURCE_MESSAGE)).toBeVisible();
        });
    });

    describe('EnableProject', () => {
        it('renders when another project is active and no media item selected', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: { project_id: '456' },
            });

            expect(await screen.findByRole('button', { name: /Activate project/i })).toBeVisible();
        });

        it('does not render EnableProject when current project is active', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: { project_id: '123' },
            });

            expect(screen.queryByRole('button', { name: /Activate project/i })).not.toBeInTheDocument();
        });

        it('does not render EnableProject when no active pipeline', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: null,
            });

            await waitFor(() => {
                expect(screen.queryByRole('button', { name: /Activate project/i })).not.toBeInTheDocument();
            });
        });
    });

    describe('StreamContainer', () => {
        it('renders when no media item selected and source/sink are configured', async () => {
            renderApp({
                pipelineConfig: { status: 'idle' },
                webRtcConfig: { status: 'idle' },
            });

            expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        });

        it('renders when no media item selected and no other project is active', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: null,
            });

            expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        });

        it('renders when current project is active', async () => {
            renderApp({
                webRtcConfig: { status: 'idle' },
                pipelineConfig: { project_id: '123', status: 'running' },
                activePipelineConfig: { project_id: '123' },
            });

            expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        });
    });

    describe('InferenceResult', () => {
        it('renders when media item is selected', async () => {
            renderApp({ selectedMediaItem: mockMediaItem });

            await waitFor(() => {
                expect(screen.queryByText(SOURCE_MESSAGE)).not.toBeInTheDocument();
                expect(screen.queryByRole('button', { name: /Start stream/i })).not.toBeInTheDocument();
            });
        });

        it('renders InferenceResult even when source/sink missing if media item selected', async () => {
            renderApp({
                pipelineConfig: { source: undefined, sink: undefined },
                selectedMediaItem: mockMediaItem,
            });

            await waitFor(() => {
                expect(screen.queryByText(SOURCE_MESSAGE)).not.toBeInTheDocument();
            });
        });

        it('renders InferenceResult even when another project is active if media item selected', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: { project_id: '456' },
                selectedMediaItem: mockMediaItem,
            });

            await waitFor(() => {
                expect(screen.queryByRole('button', { name: /Activate project/i })).not.toBeInTheDocument();
            });
        });
    });
});
