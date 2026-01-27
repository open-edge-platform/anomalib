// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TestProviders } from 'src/providers';

import { useConnectSourceToPipeline } from '../../../../../hooks/use-pipeline.hook';
import { useSourceMutation } from '../hooks/use-source-mutation.hook';
import { IpCameraFields } from '../ip-camera/ip-camera-fields.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from '../ip-camera/utils';
import { IPCameraSourceConfig } from '../util';
import { AddSource } from './add-source.component';

vi.mock('../hooks/use-source-mutation.hook');
vi.mock('../../../../../hooks/use-pipeline.hook');

describe('add-source', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const newConfig = {
        id: 'test-source-id',
        name: 'Test Camera',
        stream_url: 'rtsp://192.168.1.100:554/stream',
        source_type: 'ip_camera',
        auth_required: false,
        project_id: '123',
    };

    it('calls connectToPipelineMutation after successful submit', async () => {
        const mockOnSaved = vi.fn();
        const mockSourceMutation = vi.fn().mockResolvedValue(newConfig.id);
        const mockConnectToPipeline = vi.fn().mockResolvedValue(undefined);

        vi.mocked(useConnectSourceToPipeline).mockReturnValue(mockConnectToPipeline);
        vi.mocked(useSourceMutation).mockReturnValue(mockSourceMutation);

        render(
            <TestProviders>
                <AddSource
                    onSaved={mockOnSaved}
                    config={getIpCameraInitialConfig(newConfig.project_id)}
                    componentFields={(state: IPCameraSourceConfig) => <IpCameraFields defaultState={state} />}
                    bodyFormatter={ipCameraBodyFormatter}
                />
            </TestProviders>
        );

        const nameInput = screen.getByRole('textbox', { name: /Name/i });
        const streamUrlInput = screen.getByRole('textbox', { name: /Stream Url/i });

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, newConfig.name);
        await userEvent.clear(streamUrlInput);
        await userEvent.type(streamUrlInput, newConfig.stream_url);
        await userEvent.click(screen.getByRole('button', { name: /Add & Connect/i }));

        await waitFor(() => {
            expect(mockSourceMutation).toHaveBeenCalledWith(expect.objectContaining({ ...newConfig, id: '' }));
            expect(mockConnectToPipeline).toHaveBeenCalledWith(newConfig.id);
            expect(mockOnSaved).toHaveBeenCalled();
        });
    });
});
