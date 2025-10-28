// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Button, Flex, Form, Switch, TextField } from '@geti/ui';
import { isEmpty, isFunction } from 'lodash-es';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { IPCameraSourceConfig } from '../util';

type IpCameraProps = {
    config?: IPCameraSourceConfig;
    renderButtons?: (isPending: boolean) => ReactNode;
};

const initConfig: IPCameraSourceConfig = {
    id: '',
    name: '',
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
};

export const IpCamera = ({ config = initConfig, renderButtons }: IpCameraProps) => {
    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            source_type: 'ip_camera',
            stream_url: String(formData.get('stream_url')),
            auth_required: String(formData.get('auth_required')) === 'on' ? true : false,
        }),
    });

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />
                <TextField width={'100%'} label='Stream Url:' name='stream_url' defaultValue={state.stream_url} />
                <Switch
                    name='auth_required'
                    aria-label='Require Authentication'
                    defaultSelected={state?.auth_required}
                    key={state?.auth_required ? 'true' : 'false'}
                >
                    Require Authentication
                </Switch>

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
