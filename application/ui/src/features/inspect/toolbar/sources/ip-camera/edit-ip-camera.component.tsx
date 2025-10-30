// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Button, ButtonGroup, Divider, Form, View } from '@geti/ui';
import { useConnectSourceToPipeline } from 'src/hooks/use-pipeline.hook';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { IPCameraSourceConfig } from '../util';
import { IpCameraFields } from './ip-camera-fileds.component';

import classes from './edit-ip-camera.module.scss';

type IpCameraProps = {
    config: IPCameraSourceConfig;
    onSaved: () => void;
};

export const EditIpCamera = ({ config, onSaved }: IpCameraProps) => {
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: false,
        onSaved: async (sourceId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sourceId));
            connectToPipeline.current = false;
            onSaved();
        },
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
            <View UNSAFE_className={classes.container}>
                <IpCameraFields state={state} />
            </View>

            <Divider size='S' marginY={'size-200'} />

            <ButtonGroup marginTop={'0px'}>
                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = false)}
                >
                    Save
                </Button>

                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = true)}
                >
                    Save & Connect
                </Button>
            </ButtonGroup>
        </Form>
    );
};
