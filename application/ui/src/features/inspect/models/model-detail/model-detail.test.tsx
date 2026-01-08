// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
import { getMockedPagination } from 'mocks/mock-pagination';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { StatusBarProvider } from 'src/features/inspect/footer/status-bar/status-bar-context';
import type { ModelData } from 'src/hooks/utils';
import { TestProviders } from 'src/providers';
import { queryClient } from 'src/query-client/query-client';

import { ModelDetail } from './model-detail.component';

const projectId = 'test-project-id';

vi.mock('src/hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => ({ projectId }),
}));

const createMockModel = (overrides: Partial<ModelData> = {}): ModelData => ({
    id: 'model-1',
    name: 'PatchCore',
    status: 'Completed',
    architecture: 'PatchCore',
    timestamp: 'Dec 18, 2025, 10:30 AM',
    backbone: 'resnet50',
    startTime: Date.now() - 3600000,
    progress: 100,
    durationInSeconds: 120,
    sizeBytes: 52428800,
    job: undefined,
    ...overrides,
});

const mockedProject = getMockedProject();
const mockedPipeline = getMockedPipeline();

const TestWrapper = ({ children }: { children: ReactNode }) => (
    <TestProviders routerProps={{ initialEntries: [`/projects/${projectId}`] }}>
        <StatusBarProvider>{children}</StatusBarProvider>
    </TestProviders>
);

describe('ModelDetail', () => {
    beforeAll(() => {
        queryClient.setQueryData(
            ['get', '/api/projects/{project_id}', { params: { path: { project_id: projectId } } }],
            mockedProject
        );
        queryClient.setQueryData(['get', '/api/projects'], {
            projects: [mockedProject],
            ...getMockedPagination(),
        });
        queryClient.setQueryData(
            ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            mockedPipeline
        );
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('Model Information', () => {
        it('displays model name', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getAllByText('PatchCore').length).toBeGreaterThan(0);
        });

        it('displays Active badge for active model', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={true} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getByText('Active')).toBeInTheDocument();
        });

        it('does not display Active badge for non-active model', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.queryByText('Active')).not.toBeInTheDocument();
        });

        it('displays model information grid', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getByText('Training Date')).toBeInTheDocument();
            expect(screen.getByText('Model Size')).toBeInTheDocument();
            expect(screen.getByText('Training Duration')).toBeInTheDocument();
            expect(screen.getByText('Architecture')).toBeInTheDocument();
        });
    });

    describe('Navigation', () => {
        it('displays back button', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getByText('Back to Models')).toBeInTheDocument();
        });

        it('calls onBack when back button is clicked', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            fireEvent.click(screen.getByText('Back to Models'));
            expect(onBack).toHaveBeenCalledTimes(1);
        });
    });

    describe('Export Section', () => {
        it('displays export format options', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getByText('Export Format')).toBeInTheDocument();
            expect(screen.getByRole('radiogroup', { name: 'Select export format' })).toBeInTheDocument();
        });

        it('displays Export button', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            expect(screen.getByRole('button', { name: 'Export' })).toBeInTheDocument();
        });

        it('shows compression options when OpenVINO format is selected', () => {
            const model = createMockModel();
            const onBack = vi.fn();

            render(
                <TestWrapper>
                    <ModelDetail model={model} isActiveModel={false} onBack={onBack} />
                </TestWrapper>
            );

            // OpenVINO is selected by default
            expect(screen.getAllByText('Compression (optional)').length).toBeGreaterThan(0);
        });
    });
});
