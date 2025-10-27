// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { VideoFileSourceConfig } from '../util';
import { VideoFile } from './video-file.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-source-action.hook');

const mockedConfig: VideoFileSourceConfig = {
    id: '1',
    name: 'Test Folder',
    video_path: './test/123',
    source_type: 'video_file',
};

describe('VideoFile', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<VideoFile />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', async () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<VideoFile config={mockedConfig} />);

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => expect(mockedSubmitAction).toHaveBeenCalled());
    });

    it('renders fields with correct values from config', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<VideoFile config={mockedConfig} />);

        expect(useSourceAction).toHaveBeenCalledWith({
            config: mockedConfig,
            isNewSource: false,
            bodyFormatter: expect.anything(),
        });

        expect(screen.getByRole('textbox', { name: /^Id$/i, hidden: true })).toHaveValue(mockedConfig.id);
        expect(screen.getByRole('textbox', { name: /Name/i })).toHaveValue(mockedConfig.name);
        expect(screen.getByRole('textbox', { name: /Video file path/i })).toHaveValue(mockedConfig.video_path);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
    });
});
