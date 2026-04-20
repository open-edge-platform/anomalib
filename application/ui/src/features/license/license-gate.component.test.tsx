// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti/ui/theme';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { LicenseGate } from './license-gate.component';

const renderGate = () => {
    return render(
        <QueryClientProvider client={new QueryClient()}>
            <ThemeProvider>
                <LicenseGate>
                    <div>Studio content</div>
                </LicenseGate>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

describe('LicenseGate', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('blocks the app until licenses are accepted', async () => {
        let accepted = false;

        vi.spyOn(global, 'fetch').mockImplementation(async (input, init) => {
            const url = String(input);
            const method = init?.method ?? 'GET';

            if (url.endsWith('/api/system/license') && method === 'GET') {
                return new Response(
                    JSON.stringify({
                        accepted,
                        accepted_version: accepted ? '1.2.3' : null,
                        app_version: '1.2.3',
                        deployment_type: 'docker',
                        licenses: [
                            {
                                name: 'Apache 2.0 License',
                                url: 'https://www.apache.org/licenses/LICENSE-2.0',
                                required_for: 'Docker and development deployments',
                            },
                        ],
                    }),
                    { status: 200, headers: { 'Content-Type': 'application/json' } }
                );
            }

            if (url.endsWith('/api/system/license:accept') && method === 'POST') {
                accepted = true;
                return new Response(JSON.stringify({ accepted: true, accepted_version: '1.2.3' }), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                });
            }

            return new Response(null, { status: 404 });
        });

        renderGate();

        expect(await screen.findByText('Review licenses for Anomalib Studio 1.2.3')).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Accept and continue' }));

        await waitFor(() => {
            expect(screen.queryByText('Review licenses for Anomalib Studio 1.2.3')).not.toBeInTheDocument();
        });
    });

    it('keeps blocking when accept succeeds but refetched status is still not accepted', async () => {
        let acceptRequests = 0;

        vi.spyOn(global, 'fetch').mockImplementation(async (input, init) => {
            const url = String(input);
            const method = init?.method ?? 'GET';

            if (url.endsWith('/api/system/license') && method === 'GET') {
                return new Response(
                    JSON.stringify({
                        accepted: false,
                        accepted_version: null,
                        app_version: '1.2.3',
                        deployment_type: 'docker',
                        licenses: [
                            {
                                name: 'Apache 2.0 License',
                                url: 'https://www.apache.org/licenses/LICENSE-2.0',
                                required_for: 'Docker and development deployments',
                            },
                        ],
                    }),
                    { status: 200, headers: { 'Content-Type': 'application/json' } }
                );
            }

            if (url.endsWith('/api/system/license:accept') && method === 'POST') {
                acceptRequests += 1;
                return new Response(JSON.stringify({ accepted: true, accepted_version: '1.2.3' }), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                });
            }

            return new Response(null, { status: 404 });
        });

        renderGate();

        expect(await screen.findByText('Review licenses for Anomalib Studio 1.2.3')).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Accept and continue' }));

        await waitFor(() => {
            expect(acceptRequests).toBe(1);
        });

        expect(screen.getByText('Review licenses for Anomalib Studio 1.2.3')).toBeInTheDocument();
    });
});
