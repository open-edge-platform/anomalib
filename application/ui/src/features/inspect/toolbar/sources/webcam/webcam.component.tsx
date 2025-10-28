// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Button, Flex, Form, TextField } from '@geti/ui';
import { isEmpty, isFunction } from 'lodash-es';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { isOnlyDigits, WebcamSourceConfig } from '../util';

type WebcamProps = {
    config?: WebcamSourceConfig;
    renderButtons?: (isPending: boolean) => ReactNode;
};

const initConfig: WebcamSourceConfig = {
    id: '',
    name: '',
    source_type: 'webcam',
    device_id: 0,
};

export const Webcam = ({ config = initConfig, renderButtons }: WebcamProps) => {
    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            source_type: 'webcam',
            device_id: Number(formData.get('device_id')),
        }),
    });

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <TextField
                    width='100%'
                    label='Webcam device id'
                    name='device_id'
                    defaultValue={String(state?.device_id)}
                    validate={(value) => (isOnlyDigits(value) ? '' : 'Only digits are allowed')}
                />

                {isFunction(renderButtons) ? (
                    renderButtons(isPending)
                ) : (
                    <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                        Add & Connect
                    </Button>
                )}
            </Flex>
        </Form>
    );
};
