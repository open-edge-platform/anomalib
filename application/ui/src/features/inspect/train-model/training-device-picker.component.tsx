// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { Heading, InlineAlert, Item, Link, Picker, Text } from '@geti/ui';

import { getDeviceMetadata, selectPreferredDevice } from './utils/device-metadata';

interface UseTrainingDeviceResult {
    selectedDevice: string | null;
    setSelectedDevice: (device: string | null) => void;
    devices: string[];
}

export const useTrainingDevice = (): UseTrainingDeviceResult => {
    const { data: availableDevices } = $api.useSuspenseQuery('get', '/api/system/devices/training');
    const devices = (availableDevices ?? []).map((device) => device.type.toUpperCase());

    const [selectedDevice, setSelectedDevice] = useState<string | null>(() => {
        if (devices.length === 0) {
            return null;
        }
        return selectPreferredDevice(devices) ?? devices[0];
    });

    return {
        selectedDevice,
        setSelectedDevice,
        devices,
    };
};

interface TrainingDevicePickerProps {
    selectedDevice: string | null;
    onDeviceChange: (device: string | null) => void;
    devices: string[];
}

export const TrainingDevicePicker = ({ selectedDevice, onDeviceChange, devices }: TrainingDevicePickerProps) => {
    const handleDeviceChange = (key: Key | null) => {
        onDeviceChange(key === null ? null : String(key));
    };

    if (devices.length === 0) {
        return (
            <InlineAlert variant='notice'>
                <Heading level={5}>No training devices detected</Heading>
                <Text>
                    Geti Inspect was unable to discover any compatible training hardware. If you believe this is an
                    error,{' '}
                    <Link
                        isQuiet
                        href='https://github.com/open-edge-platform/anomalib/issues'
                        target='_blank'
                        rel='noreferrer noopener'
                    >
                        let us know on GitHub
                    </Link>
                    .
                </Text>
            </InlineAlert>
        );
    }

    return (
        <Picker
            aria-label='Select a training device'
            selectedKey={selectedDevice}
            onSelectionChange={handleDeviceChange}
            width='size-3400'
            items={devices.map((device) => {
                const meta = getDeviceMetadata(device);
                return { id: device, ...meta };
            })}
        >
            {(item) => (
                <Item key={item.id} textValue={item.label}>
                    <Text>{item.label}</Text>
                    {item.description ? <Text slot='description'>{item.description}</Text> : null}
                </Item>
            )}
        </Picker>
    );
};
