// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { ActionButton, Flex, Item, Key, Loading, Picker, TextField } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { WebcamSourceConfig } from '../util';

type WebcamFieldsProps = {
    defaultState: WebcamSourceConfig;
};

export const WebcamFields = ({ defaultState }: WebcamFieldsProps) => {
    const { data: cameraDevices, isLoading, isRefetching, refetch } = $api.useQuery('get', '/api/devices/camera');
    const [name, setName] = useState(defaultState.name);
    const [isModified, setIsModified] = useState(false);

    const devices = (cameraDevices?.devices ?? []).map((device) => ({
        id: device.index,
        name: device.name,
    }));

    const handleNameChange = (value: string) => {
        setName(value);
        setIsModified(true);
    };

    const handleSelectionChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        const device = devices.find((d) => d.id === Number(key));

        // if user modifies the name field, don't override it
        if (device && (!isModified || !name?.trim())) {
            setName(device.name);
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isHidden label='name' name='name' value={name} />
            <TextField width={'100%'} label='Name' name='name_display' value={name} onChange={handleNameChange} />

            <Flex alignItems='end' gap='size-200'>
                <Picker
                    flex='1'
                    label='Camera'
                    name='device_id'
                    items={devices}
                    isLoading={isLoading}
                    defaultSelectedKey={defaultState.device_id}
                    onSelectionChange={handleSelectionChange}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <ActionButton
                    onPress={() => refetch()}
                    isQuiet
                    aria-label='Refresh Cameras'
                    isDisabled={isLoading || isRefetching}
                >
                    {isRefetching ? <Loading mode={'inline'} size='S' /> : <Refresh />}
                </ActionButton>
            </Flex>
        </Flex>
    );
};
