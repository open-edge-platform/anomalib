// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Form } from '@geti/ui';
import { useConnectSourceToPipeline } from 'src/hooks/use-pipeline.hook';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { IPCameraSourceConfig } from '../util';
import { IpCameraFields } from './ip-camera-fileds.component';

type AddIpCameraProps = {
    onSaved: () => void;
};

const initConfig: IPCameraSourceConfig = {
    id: '',
    name: '',
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
};

export const AddIpCamera = ({ onSaved }: AddIpCameraProps) => {
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config: initConfig,
        isNewSource: true,
        onSaved: async (sourceId) => {
            await connectToPipelineMutation(sourceId);
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
            <IpCameraFields state={state} />

            <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                Add & Connect
            </Button>
        </Form>
    );
};
