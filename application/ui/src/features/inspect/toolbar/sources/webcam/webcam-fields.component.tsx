// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, TextField } from '@geti/ui';

import { isOnlyDigits, WebcamSourceConfig } from '../util';

type WebcamFieldsProps = {
    state: WebcamSourceConfig;
};

export const WebcamFields = ({ state }: WebcamFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={state.id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={state.name} />

            <TextField
                width='100%'
                label='Webcam device id'
                name='device_id'
                defaultValue={String(state.device_id)}
                validate={(value) => (isOnlyDigits(value) ? '' : 'Only digits are allowed')}
            />
        </Flex>
    );
};
