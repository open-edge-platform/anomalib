/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ProjectEdition } from './project-edition.component';

describe('ProjectEdition', () => {
    const defaultProps = {
        name: 'Test Project',
        isPending: false,
        onChange: vi.fn(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders with the initial project name', () => {
        render(<ProjectEdition {...defaultProps} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i });
        expect(input).toHaveValue('Test Project');
    });

    it('auto-selects the text field on mount', async () => {
        render(<ProjectEdition {...defaultProps} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i }) as HTMLInputElement;

        await waitFor(() => {
            expect(input.selectionStart).toBe(0);
            expect(input.selectionEnd).toBe(defaultProps.name.length);
        });
    });

    it('calls onBlur with new name when input loses focus', async () => {
        const onChange = vi.fn();

        render(<ProjectEdition {...defaultProps} onChange={onChange} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i });

        await userEvent.clear(input);
        await userEvent.type(input, 'New Project Name');
        await userEvent.tab();

        expect(onChange).toHaveBeenCalledWith('New Project Name');
    });

    it('calls onBlur with new name when Enter key is pressed', async () => {
        const onChange = vi.fn();

        render(<ProjectEdition {...defaultProps} onChange={onChange} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i });

        await userEvent.clear(input);
        await userEvent.type(input, 'New Project Name');
        await userEvent.keyboard('{Enter}');

        expect(onChange).toHaveBeenCalledWith('New Project Name');
    });

    it('resets to original name and calls onBlur when Escape key is pressed', async () => {
        const onChange = vi.fn();

        render(<ProjectEdition {...defaultProps} onChange={onChange} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i });

        await userEvent.clear(input);
        await userEvent.type(input, 'Modified Name');
        await userEvent.keyboard('{Escape}');

        expect(input).toHaveValue('Test Project');
        expect(onChange).toHaveBeenCalledWith('Test Project');
    });

    it('disables the input when isPending is true', () => {
        render(<ProjectEdition {...defaultProps} isPending={true} />);

        const input = screen.getByRole('textbox', { name: /edit project name/i });
        expect(input).toBeDisabled();
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
});
