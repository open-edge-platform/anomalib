// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DialogContainer } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TRAINABLE_MODELS } from 'mocks/mock-trainable-models';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { TrainModelDialog } from './train-model-dialog.component';

type DeviceType = 'cpu' | 'xpu' | 'cuda';

interface MockDeviceInfo {
    type: DeviceType;
    name: string;
    memory: number | null;
    index: number | null;
}

const DEFAULT_MOCK_DEVICES: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null },
    { type: 'cuda', name: 'NVIDIA GPU', memory: 8589934592, index: 0 },
];

describe('TrainModelDialog', () => {
    const closeMock = vi.fn();

    const renderDialog = ({ devices = DEFAULT_MOCK_DEVICES }: { devices?: MockDeviceInfo[] } = {}) => {
        server.use(
            http.get('/api/trainable-models', ({ response }) =>
                response(200).json({ trainable_models: TRAINABLE_MODELS })
            ),
            http.get('/api/system/devices/training', ({ response }) => response(200).json(devices)),
            http.post('/api/jobs:train', ({ response }) => response(200).json({ job_id: 'job-123' }))
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route
                                path='/projects/:projectId/inspect'
                                element={
                                    <DialogContainer onDismiss={closeMock}>
                                        <TrainModelDialog close={closeMock} />
                                    </DialogContainer>
                                }
                            />
                        </Routes>
                    </MemoryRouter>
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    beforeEach(() => {
        closeMock.mockClear();
    });

    describe('Form Validation', () => {
        it('disables Start button when no model is selected', async () => {
            renderDialog();

            expect(await screen.findByText('Train model')).toBeVisible();

            const startButton = screen.getByRole('button', { name: /start/i });
            expect(startButton).toBeDisabled();
        });

        it('disables Start button when no device is selected', async () => {
            renderDialog({ devices: [] });

            expect(await screen.findByText('Train model')).toBeVisible();

            const startButton = screen.getByRole('button', { name: /start/i });
            expect(startButton).toBeDisabled();
        });

        it('enables Start button when model and device are selected', async () => {
            renderDialog();

            expect(await screen.findByText('PatchCore')).toBeVisible();

            await userEvent.click(screen.getByText('PatchCore'));

            await waitFor(() => {
                const startButton = screen.getByRole('button', { name: /start/i });
                expect(startButton).not.toBeDisabled();
            });
        });
    });

    describe('Form Submission', () => {
        it('closes dialog after successful submission', async () => {
            renderDialog();

            expect(await screen.findByText('PatchCore')).toBeVisible();
            await userEvent.click(screen.getByText('PatchCore'));

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /start/i })).not.toBeDisabled();
            });

            await userEvent.click(screen.getByRole('button', { name: /start/i }));

            await waitFor(() => {
                expect(closeMock).toHaveBeenCalled();
            });
        });
    });
});
