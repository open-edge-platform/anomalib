// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { getMockedMediaItem } from 'mocks/mock-media-item';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { describe, expect, it } from 'vitest';

import { ProjectThumbnail } from './project-thumbnail.component';

const renderWithProviders = (ui: React.ReactElement) => {
    return render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>);
};

describe('ProjectThumbnail', () => {
    afterEach(() => {
        server.resetHandlers();
        cleanup();
    });

    it('displays image thumbnail when project has images', async () => {
        server.use(
            http.get('/api/projects/{project_id}/images', () => {
                return HttpResponse.json({
                    media: [getMockedMediaItem({ id: 'image-1', project_id: 'project-123' })],
                    pagination: { offset: 0, limit: 1, count: 1, total: 1 },
                });
            })
        );

        renderWithProviders(<ProjectThumbnail projectId='project-123' projectName='Test Project' />);

        await waitFor(() => {
            expect(screen.getByRole('img', { name: /Test Project thumbnail/i })).toBeVisible();
        });
    });

    it('displays placeholder when project has no images', async () => {
        server.use(
            http.get('/api/projects/{project_id}/images', () => {
                return HttpResponse.json({
                    media: [],
                    pagination: { offset: 0, limit: 1, count: 0, total: 0 },
                });
            })
        );

        renderWithProviders(<ProjectThumbnail projectId='project-123' projectName='Test Project' />);

        await waitFor(() => {
            expect(screen.queryByRole('img', { name: /thumbnail/i })).not.toBeInTheDocument();
            expect(screen.getByText('T')).toBeVisible();
        });
    });

    it('displays placeholder when projectId is not provided', async () => {
        renderWithProviders(<ProjectThumbnail projectName='Test Project' />);

        await waitFor(() => {
            expect(screen.queryByRole('img', { name: /thumbnail/i })).not.toBeInTheDocument();
            expect(screen.getByText('T')).toBeVisible();
        });
    });
});
