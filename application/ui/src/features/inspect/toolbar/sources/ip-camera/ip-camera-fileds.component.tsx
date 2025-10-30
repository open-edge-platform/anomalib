// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Switch, TextField } from '@geti/ui';

import { IPCameraSourceConfig } from '../util';

type IpCameraFieldsProps = {
    state: IPCameraSourceConfig;
};

export const IpCameraFields = ({ state }: IpCameraFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={state.id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={state.name} />
            <TextField width={'100%'} label='Stream Url:' name='stream_url' defaultValue={state.stream_url} />
            <Switch
                name='auth_required'
                aria-label='Require Authentication'
                defaultSelected={state.auth_required}
                key={state.auth_required ? 'true' : 'false'}
            >
                Require Authentication
            </Switch>
        </Flex>
    );
};
