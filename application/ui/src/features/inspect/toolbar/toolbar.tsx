// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, View } from '@geti/ui';

import { InferenceDevices } from './inference-devices/inference-devices.component';
import { PipelineConfiguration } from './pipeline-configuration.component';

export const Toolbar = () => {
    return (
        <View
            gridArea='toolbar'
            padding='size-200'
            backgroundColor={'gray-100'}
            UNSAFE_style={{ fontSize: dimensionValue('size-150'), color: 'var(--spectrum-global-color-gray-800)' }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'}>
                <Flex marginStart='auto' alignItems={'center'} gap={'size-200'}>
                    <Divider size={'S'} orientation={'vertical'} />
                    <InferenceDevices />
                    <Divider size={'S'} orientation={'vertical'} />
                    <PipelineConfiguration />
                </Flex>
            </Flex>
        </View>
    );
};
