// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, TextField } from '@geti/ui';

import { isOnlyDigits, WebcamSourceConfig } from '../util';

type WebcamFieldsProps = {
    defaultState: WebcamSourceConfig;
};

export const WebcamFields = ({ defaultState }: WebcamFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={defaultState.name} />

            <TextField
                width='100%'
                label='Webcam device id'
                name='device_id'
                defaultValue={String(defaultState.device_id)}
                validate={(value) => (isOnlyDigits(value) ? '' : 'Only digits are allowed')}
            />
        </Flex>
    );
};
