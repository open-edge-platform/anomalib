// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { MediaThumbnail } from './media-thumbnail.component';

describe('MediaThumbnail', () => {
    it('calls onClick when image is clicked', async () => {
        const mockedClick = vi.fn();
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' onClick={mockedClick} />);

        userEvent.click(screen.getByRole('img', { name: 'Test Image' }));
        await waitFor(() => expect(mockedClick).toHaveBeenCalled());
    });

    it('calls onDoubleClick when image is double-clicked', async () => {
        const mockedDblClick = vi.fn();
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' onDoubleClick={mockedDblClick} />);

        userEvent.dblClick(screen.getByRole('img', { name: 'Test Image' }));
        await waitFor(() => expect(mockedDblClick).toHaveBeenCalled());
    });
});
