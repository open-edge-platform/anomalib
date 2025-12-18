import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TestProviders } from 'src/providers';

import { ProjectActions } from './project-list-actions.component';

describe('ProjectActions', () => {
    it('displays rename option in the menu', async () => {
        render(
            <TestProviders>
                <ProjectActions onRename={vi.fn()} />
            </TestProviders>
        );

        await userEvent.click(screen.getByRole('button'));

        expect(screen.getByRole('menuitem', { name: /Rename/i })).toBeVisible();
    });

    it('calls onRename when rename menu item is clicked', async () => {
        const mockOnRename = vi.fn();
        render(
            <TestProviders>
                <ProjectActions onRename={mockOnRename} />
            </TestProviders>
        );

        await userEvent.click(screen.getByRole('button'));
        await userEvent.click(screen.getByRole('menuitem', { name: /Rename/i }));

        expect(mockOnRename).toHaveBeenCalledTimes(1);
    });
});
