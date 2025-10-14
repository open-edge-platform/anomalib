// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { StatusLight } from '@adobe/react-spectrum';
import { Button, Divider, Flex, Item, Picker, View } from '@geti/ui';
import { isString, uniq } from 'lodash-es';

import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';
import { useProjectTrainingJobs } from './hooks/use-project-training-jobs.hook';
import { useInference } from './inference-provider.component';

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

const ModelsPicker = () => {
    const { jobs } = useProjectTrainingJobs();
    const { selectedModelId, onSetSelectedModelId } = useInference();

    const models = uniq(
        jobs?.map((job) => job.payload.model_name).filter((name) => name !== undefined && isString(name))
    ).map((name) => ({
        id: name,
        name,
    }));

    useEffect(() => {
        if (selectedModelId !== undefined || models.length === 0) {
            return;
        }

        onSetSelectedModelId(models[0].id);
    }, [selectedModelId, models, onSetSelectedModelId]);

    if (jobs === undefined || jobs.length === 0) {
        return null;
    }

    if (models.length === 0) {
        return null;
    }

    return (
        <Picker
            items={models}
            label={'Model'}
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

                <Flex marginStart='auto'>
                    <ModelsPicker />
                </Flex>
            </Flex>
        </View>
    );
};
