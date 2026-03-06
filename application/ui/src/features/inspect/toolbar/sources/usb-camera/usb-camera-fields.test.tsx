// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';

import { UsbCameraSourceConfig } from '../util';
import { UsbCameraFields } from './usb-camera-fields.component';

const getMockState = (props: Partial<UsbCameraSourceConfig> = {}): UsbCameraSourceConfig => ({
    id: 'test-id',
    name: 'Test Camera',
    device_id: 0,
    project_id: 'test-project',
    source_type: 'usb_camera' as const,
    ...props,
});

describe('UsbCameraFields', () => {
    const mockCameraDevices = [
        { index: 0, name: 'Camera 0' },
        { index: 1, name: 'Camera 1' },
        { index: 2, name: 'Camera 2' },
    ];

    const renderComponent = (state: UsbCameraSourceConfig) => {
        server.use(
            http.get('/api/system/devices/camera', () => {
                return HttpResponse.json(mockCameraDevices);
            })
        );

        return render(
            <TestProviders>
                <UsbCameraFields defaultState={state} />
            </TestProviders>
        );
    };

    it('displays the default name value', () => {
        const name = 'Test Camera';
        renderComponent(getMockState({ name }));

        expect(screen.getByRole('textbox', { name: /name/i })).toHaveValue(name);
    });

    it('loads and displays camera devices', async () => {
        renderComponent(getMockState());

        const cameraButton = await screen.findByRole('button', { name: /Camera list/i });
        await userEvent.click(cameraButton);

        for (const device of mockCameraDevices) {
            expect(await screen.findByRole('option', { name: device.name })).toBeInTheDocument();
        }
    });

    it('updates name when user types', async () => {
        const newName = 'New Camera Name';
        renderComponent(getMockState());

        const nameInput = screen.getByRole('textbox', { name: /name/i });
        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, newName);

        expect(nameInput).toHaveValue(newName);
    });

    it('updates name with device name when name is system-generated', async () => {
        const selectedOption = mockCameraDevices[1];
        renderComponent(getMockState({ name: '' }));

        const cameraButton = await screen.findByRole('button', { name: /Camera list/i });
        await userEvent.click(cameraButton);

        const camera2Option = await screen.findByRole('option', { name: selectedOption.name });
        await userEvent.click(camera2Option);

        await waitFor(() => {
            const nameInput = screen.getByRole('textbox', { name: /name/i });
            expect(nameInput).toHaveValue(selectedOption.name);
        });
    });

    it('does not update name with device name when user has manually changed it', async () => {
        const newName = 'Custom Name';
        renderComponent(getMockState());

        const nameInput = screen.getByRole('textbox', { name: /name/i });
        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, newName);

        const cameraButton = await screen.findByRole('button', { name: /Camera list/i });
        await userEvent.click(cameraButton);

        const camera1Option = await screen.findByRole('option', { name: 'Camera 1' });
        await userEvent.click(camera1Option);

        await waitFor(() => {
            expect(nameInput).toHaveValue(newName);
        });
    });
});
