// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { ImagesFolderSourceConfig } from '../util';
import { ImageFolder } from './image-folder.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-source-action.hook');

const mockedConfig: ImagesFolderSourceConfig = {
    id: '1',
    name: 'Test Folder',
    source_type: 'images_folder',
    images_folder_path: '/path/to/images',
    ignore_existing_images: true,
};

describe('ImageFolder', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<ImageFolder />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', async () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<ImageFolder config={mockedConfig} />);

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => expect(mockedSubmitAction).toHaveBeenCalled());
    });

    it('renders fields with correct values from config', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<ImageFolder config={mockedConfig} />);

        expect(useSourceAction).toHaveBeenCalledWith({
            config: mockedConfig,
            isNewSource: false,
            bodyFormatter: expect.anything(),
        });

        expect(screen.getByRole('textbox', { name: /^Id$/i, hidden: true })).toHaveValue(mockedConfig.id);
        expect(screen.getByRole('textbox', { name: /Name/i })).toHaveValue(mockedConfig.name);
        expect(screen.getByRole('textbox', { name: /Images folder path/i })).toHaveValue(
            mockedConfig.images_folder_path
        );
        expect(screen.getByLabelText(/ignore existing images/i)).toBeChecked();

        expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
    });
});
