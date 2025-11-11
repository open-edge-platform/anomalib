// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Divider, Flex, Item, Picker, StatusLight, View } from '@geti/ui';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { useInference } from '../inference-provider.component';
import { InferenceOpacity } from './inference-opacity';
import { InputOutputSetup } from './pipeline-configuration.component';
import { PipelineSwitch } from './pipeline-switch/pipeline-switch.component';

const WebRTCConnectionStatus = () => {
    const { status, stop } = useWebRTCConnection();

    switch (status) {
        case 'idle':
            return (
                <Flex
                    gap='size-100'
                    alignItems={'center'}
                    UNSAFE_style={{
                        '--spectrum-gray-visual-color': 'var(--spectrum-global-color-gray-500)',
                    }}
                >
                    <StatusLight role={'status'} aria-label='Idle' variant='neutral'>
                        Idle
                    </StatusLight>
                </Flex>
            );
        case 'connecting':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connecting' variant='info'>
                        Connecting
                    </StatusLight>
                </Flex>
            );
        case 'disconnected':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Disconnected' variant='negative'>
                        Disconnected
                    </StatusLight>
                </Flex>
            );
        case 'failed':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Failed' variant='negative'>
                        Failed
                    </StatusLight>
                </Flex>
            );
        case 'connected':
            return (
                <Flex gap='size-200' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connected' variant='positive'>
                        Connected
                    </StatusLight>
                    <Button onPress={stop} variant='secondary'>
                        Stop
                    </Button>
                </Flex>
            );
    }
};

const useTrainedModels = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data?.models.map((model) => ({ id: model.id, name: model.name })) || [];
};

const ModelsPicker = () => {
    const models = useTrainedModels();
    const { selectedModelId, onSetSelectedModelId } = useInference();

    useEffect(() => {
        if (selectedModelId !== undefined || models.length === 0) {
            return;
        }

        onSetSelectedModelId(models[0].id);
    }, [selectedModelId, models, onSetSelectedModelId]);

    if (models === undefined || models.length === 0) {
        return null;
    }

    return (
        <Picker
            aria-label='Select model'
            items={models}
            selectedKey={selectedModelId}
            onSelectionChange={(key) => onSetSelectedModelId(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};

export const Toolbar = () => {
    return (
        <View
            backgroundColor={'gray-100'}
            gridArea='toolbar'
            padding='size-200'
            UNSAFE_style={{
                fontSize: '12px',
                color: 'var(--spectrum-global-color-gray-800)',
            }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'}>
                <WebRTCConnectionStatus />

                <Divider orientation='vertical' size='S' />

                <Flex marginStart='auto' alignItems={'center'} gap={'size-200'}>
                    <ModelsPicker />
                    <Divider size={'S'} orientation={'vertical'} />
                    <InferenceOpacity />
                    <Divider size={'S'} orientation={'vertical'} />
                    <InputOutputSetup />
                    <PipelineSwitch />
                </Flex>
            </Flex>
        </View>
    );
};
