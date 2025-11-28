import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { RequiredTextField } from './required-text-field.component';

describe('RequiredTextField', () => {
    const errorMessage = 'This field is required';

    it('does not display the error message before the input is interacted with', () => {
        render(<RequiredTextField errorMessage={errorMessage} aria-label='Required Text Field' />);

        expect(screen.queryByText(errorMessage)).not.toBeInTheDocument();
    });

    it('shows error message when field has been touched and is empty', async () => {
        render(<RequiredTextField errorMessage={errorMessage} aria-label='Required Text Field' />);

        const input = screen.getByLabelText(/Required Text Field/i);

        await userEvent.type(input, 'test');
        await userEvent.clear(input);

        expect(await screen.findByText(errorMessage)).toBeVisible();
    });
});
