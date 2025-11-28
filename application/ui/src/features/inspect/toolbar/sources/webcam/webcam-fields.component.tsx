// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { Flex, Item, Picker, TextField } from '@geti/ui';

import { WebcamSourceConfig } from '../util';

type WebcamFieldsProps = {
    defaultState: WebcamSourceConfig;
};

export const WebcamFields = ({ defaultState }: WebcamFieldsProps) => {
    const { data: cameraDevices, isLoading } = $api.useQuery('get', '/api/devices/camera');

    const devices = (cameraDevices?.devices ?? []).map((device, index) => ({ name: device.name, id: index }));

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={defaultState.name} />

            <Picker
                width='auto'
                label='Cameras'
                name='device_id'
                items={devices}
                isLoading={isLoading}
                defaultSelectedKey={defaultState.device_id}
            >
                {(item) => <Item>{item.name}</Item>}
            </Picker>
        </Flex>
    );
};
