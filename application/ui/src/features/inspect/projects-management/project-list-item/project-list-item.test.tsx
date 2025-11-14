import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useNavigate } from 'react-router';
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

const mockStop = vi.fn();
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
            stop: mockStop,
            start: vi.fn(),
            status: 'idle',
            webRTCConnectionRef: { current: null },
        });
    });

    it('navigates to project when clicked', async () => {
        render(<ProjectListItem project={mockProject} isInEditMode={false} onBlur={vi.fn()} isActive={false} />);

        await userEvent.click(screen.getByRole('listitem'));

        expect(mockStop).toHaveBeenCalled();
        expect(mockNavigate).toHaveBeenCalledWith('/projects/project-123?mode=Dataset');
    });
});
