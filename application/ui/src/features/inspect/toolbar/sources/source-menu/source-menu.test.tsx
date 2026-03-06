import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';

import { SourceMenu, SourceMenuProps } from './source-menu.component';

vi.mock('@anomalib-studio/hooks', () => ({ useProjectIdentifier: () => ({ projectId: '123' }) }));

describe('SourceMenu', () => {
    const renderApp = ({
        id = 'id-test',
        name = 'name test',
        isConnected = false,
        onEdit = vi.fn(),
    }: Partial<SourceMenuProps>) => {
        render(
            <TestProviders>
                <SourceMenu id={id} name={name} isConnected={isConnected} onEdit={onEdit} />
            </TestProviders>
        );
    };

    it('edit', async () => {
        const mockedOnEdit = vi.fn();

        renderApp({ onEdit: mockedOnEdit });

        await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Edit/i }));

        expect(mockedOnEdit).toHaveBeenCalled();
    });

    describe('remove', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            const pipelinePatchSpy = vi.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json({}, { status });
                }),
                http.delete('/api/projects/{project_id}/sources/{source_id}', () =>
                    HttpResponse.json(null, { status: 204 })
                )
            );

            return pipelinePatchSpy;
        };

        it('success', async () => {
            const pipelinePatchSpy = configRequests();

            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Remove/i }));

            await expect(await screen.findByLabelText('toast')).toHaveTextContent(
                `${name} has been removed successfully!`
            );
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });

        it('disabled when source is connected', async () => {
            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));

            expect(screen.getByRole('menuitem', { name: /Remove/i })).toHaveAttribute('aria-disabled', 'true');
        });
    });

    describe('connect', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            server.use(http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json({}, { status })));
        };

        it('success', async () => {
            configRequests();

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Connect/i }));

            await expect(await screen.findByLabelText('toast')).toHaveTextContent(
                `Successfully connected to "${name}"`
            );
        });

        it('error', async () => {
            configRequests(500);

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Connect/i }));

            await expect(await screen.findByLabelText('toast')).toHaveTextContent(`Failed to connect to "${name}"`);
        });
    });
});
