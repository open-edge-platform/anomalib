// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { render } from '../../../tests/utils';
import { BackendReadyGate } from './backend-ready-gate.component';

const EMPTY_PROJECTS = { projects: [], pagination: { offset: 0, limit: 10, count: 0, total: 0 } };

const renderGate = (maxRetries = 5) =>
    render(
        <BackendReadyGate retryDelayMs={1} maxRetries={maxRetries}>
            <div>App content</div>
        </BackendReadyGate>
    );

describe('BackendReadyGate', () => {
    it('renders the app once the backend responds', async () => {
        server.use(http.get('/api/projects', () => HttpResponse.json(EMPTY_PROJECTS)));

        renderGate();

        expect(await screen.findByText('App content')).toBeInTheDocument();
    });

    it('keeps waiting while the backend is not reachable yet', async () => {
        let attempts = 0;
        server.use(
            http.get('/api/projects', () => {
                attempts += 1;

                return attempts < 3 ? HttpResponse.error() : HttpResponse.json(EMPTY_PROJECTS);
            })
        );

        renderGate();

        expect(await screen.findByText('App content')).toBeInTheDocument();
        expect(attempts).toBe(3);
    });

    it('treats an HTTP error response as a running backend', async () => {
        server.use(http.get('/api/projects', () => HttpResponse.json({ detail: [] }, { status: 500 })));

        renderGate();

        expect(await screen.findByText('App content')).toBeInTheDocument();
    });

    it('shows an error with a retry button when the backend never becomes reachable', async () => {
        let reachable = false;
        server.use(
            http.get('/api/projects', () => (reachable ? HttpResponse.json(EMPTY_PROJECTS) : HttpResponse.error()))
        );

        renderGate(1);

        expect(await screen.findByText('Could not connect to the Anomalib Studio server')).toBeInTheDocument();
        expect(screen.queryByText('App content')).not.toBeInTheDocument();

        reachable = true;
        await userEvent.click(screen.getByRole('button', { name: 'Try again' }));

        expect(await screen.findByText('App content')).toBeInTheDocument();
    });
});
