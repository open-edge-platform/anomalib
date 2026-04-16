// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePipeline } from '@anomalib-studio/hooks';
import { dimensionValue, Divider, Flex, View } from '@geti/ui';
import { isNil } from 'lodash-es';

import { AnomalyMap } from './anomaly-map/anomaly-map.component';
import { InferenceDevices } from './inference-devices/inference-devices.component';
import { PipelineConfiguration } from './pipeline-configuration.component';

export const Toolbar = () => {
    const { data: pipeline } = usePipeline();

    const hasModel = !isNil(pipeline?.model?.id);

    return (
        <View
            gridArea='toolbar'
            padding='size-200'
            backgroundColor={'gray-100'}
            UNSAFE_style={{ fontSize: dimensionValue('size-150'), color: 'var(--spectrum-global-color-gray-800)' }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'} justifyContent={'space-between'}>
                <Flex gap={'size-200'}>
                    {hasModel && (
                        <>
                            <InferenceDevices />
                            <Divider size={'S'} orientation={'vertical'} />
                            <AnomalyMap />
                        </>
                    )}
                </Flex>

                <PipelineConfiguration />
            </Flex>
        </View>
    );
};
