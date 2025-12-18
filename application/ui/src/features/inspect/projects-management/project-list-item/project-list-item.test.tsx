import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useNavigate } from 'react-router';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { vi } from 'vitest';

import { useWebRTCConnection } from '../../../../components/stream/web-rtc-connection-provider';
import { Project, ProjectListItem } from './project-list-item.component';

vi.mock('../../../../components/stream/web-rtc-connection-provider');

vi.mock('react-router', async () => {
    const actual = await vi.importActual('react-router');
    return {
        ...actual,
        useNavigate: vi.fn(),
    };
});

const mockNavigate = vi.fn();

describe('ProjectListItem', () => {
    const mockProject: Project = {
        id: 'project-123',
        name: 'Test Project',
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useNavigate).mockReturnValue(mockNavigate);
        vi.mocked(useWebRTCConnection).mockReturnValue({
            stop: vi.fn(),
            start: vi.fn(),
            status: 'idle',
            webRTCConnectionRef: { current: null },
        });
    });

    it('navigates to project when clicked', async () => {
        render(
            <QueryClientProvider client={new QueryClient()}>
                <ProjectListItem
                    project={mockProject}
                    isInEditMode={false}
                    isActive={false}
                    setProjectInEdition={vi.fn()}
                />
            </QueryClientProvider>
        );

        await userEvent.click(screen.getByRole('listitem'));

        expect(mockNavigate).toHaveBeenCalledWith('/projects/project-123?mode=Dataset');
    });

    it('updates project name when edited', async () => {
        const updateNameSpy = vi.fn();
        const mockedSetProjectInEdition = vi.fn();

        server.use(
            http.patch('/api/projects/{project_id}', () => {
                updateNameSpy();
                return HttpResponse.json({});
            })
        );

        render(
            <QueryClientProvider client={new QueryClient()}>
                <ProjectListItem
                    project={mockProject}
                    isActive={true}
                    isInEditMode={true}
                    setProjectInEdition={mockedSetProjectInEdition}
                />
            </QueryClientProvider>
        );

        const input = screen.getByRole('textbox', { name: /edit project name/i });
        await userEvent.clear(input);
        await userEvent.type(input, 'Updated Project Name');
        await userEvent.tab();

        expect(updateNameSpy).toHaveBeenCalled();
        expect(mockedSetProjectInEdition).toHaveBeenCalledWith(null);
    });
});
