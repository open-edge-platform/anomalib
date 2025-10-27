// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TestProviders } from 'src/providers';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { WebcamSourceConfig } from '../util';
import { Webcam } from './webcam.component';

vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router-dom')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-source-action.hook');

const mockedConfig: WebcamSourceConfig = {
    id: '1',
    name: 'Test Folder',
    source_type: 'webcam',
    device_id: 0,
};

describe('Webcam', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(
            <TestProviders>
                <Webcam />
            </TestProviders>
        );

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', async () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(
            <TestProviders>
                <Webcam config={mockedConfig} />
            </TestProviders>
        );

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => expect(mockedSubmitAction).toHaveBeenCalled());
    });

    it('renders fields with correct values from config', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(
            <TestProviders>
                <Webcam config={mockedConfig} />
            </TestProviders>
        );

        expect(useSourceAction).toHaveBeenCalledWith({
            config: mockedConfig,
            isNewSource: false,
            bodyFormatter: expect.anything(),
        });

        expect(screen.getByRole('textbox', { name: /^Id$/i, hidden: true })).toHaveValue(mockedConfig.id);
        expect(screen.getByRole('textbox', { name: /Name/i })).toHaveValue(mockedConfig.name);
        expect(screen.getByRole('textbox', { name: /Webcam device id/i })).toHaveValue(String(mockedConfig.device_id));

        expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
    });
});
