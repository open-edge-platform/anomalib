// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, View } from '@geti/ui';

import { InferenceDevices } from '../models/inference-devices.component';
import { InferenceOpacity } from './inference-opacity';
import { InputOutputSetup } from './pipeline-configuration.component';
import { PipelineSwitch } from './pipeline-switch/pipeline-switch.component';

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
                <Flex marginStart='auto' alignItems={'center'} gap={'size-200'}>
                    <Divider size={'S'} orientation={'vertical'} />
                    <InferenceOpacity />
                    <Divider size={'S'} orientation={'vertical'} />
                    <InputOutputSetup />
                    <PipelineSwitch />
                    <Divider size={'S'} orientation={'vertical'} />
                    <InferenceDevices />
                </Flex>
            </Flex>
        </View>
    );
};
