import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { cloneDeep } from 'lodash-es';
import { MemoryRouter } from 'react-router-dom';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedMetrics } from '../../../../../mocks/mock-metrics';
import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { Fps } from './fps.component';

vi.mock('src/hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => ({ projectId: 'project-123' }),
}));

describe('Fps', () => {
    const renderFps = ({
        metricsConfig,
        pipelineConfig,
    }: {
        metricsConfig?: Partial<ReturnType<typeof getMockedMetrics>> | null;
        pipelineConfig?: Partial<ReturnType<typeof getMockedPipeline>> | null;
    } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline/metrics', ({ response }) =>
                response(200).json(getMockedMetrics(metricsConfig ? metricsConfig : {}))
            ),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig ? pipelineConfig : {}))
            )
        );
        return render(
            <QueryClientProvider client={new QueryClient()}>
                <MemoryRouter>
                    <Fps projectId={'123'} />
                </MemoryRouter>
            </QueryClientProvider>
        );
    };

    it('renders FPS value when metrics are available', async () => {
        const metricsConfig = cloneDeep(getMockedMetrics({}));
        metricsConfig.inference.latency.latest_ms = 25;

        renderFps({ metricsConfig });
        expect(await screen.findByText(/40/)).toBeVisible();
    });

    it('renders nothing if metrics are missing', async () => {
        renderFps({ pipelineConfig: { status: 'running' }, metricsConfig: {} });

        await waitFor(() => {
            expect(screen.queryByText(/FPS/i)).not.toBeInTheDocument();
        });
    });

    it('renders nothing if pipeline is not running', async () => {
        renderFps({ pipelineConfig: { status: 'active' }, metricsConfig: {} });

        await waitFor(() => {
            expect(screen.queryByText(/FPS/i)).not.toBeInTheDocument();
        });
    });
});
