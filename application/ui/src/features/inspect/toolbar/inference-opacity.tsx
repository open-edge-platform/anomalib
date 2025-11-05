// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogTrigger, Flex, Slider, Text, View } from '@geti/ui';
import { ChevronDownSmall } from '@geti/ui/icons';

import { useInference } from '../inference-provider.component';

export const InferenceOpacity = () => {
    const { inferenceOpacity, onInferenceOpacityChange, inferenceResult } = useInference();

    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <Text>Opacity:</Text>
            <DialogTrigger type={'popover'} placement={'bottom'}>
                <ActionButton width={'size-800'} isDisabled={inferenceResult === undefined}>
                    <Flex alignItems={'center'} gap={'size-50'}>
                        <span>{Math.floor(inferenceOpacity * 100)}%</span>
                        <Flex>
                            <ChevronDownSmall style={{ order: 1 }} />
                        </Flex>
                    </Flex>
                </ActionButton>
                <View padding={'size-100'}>
                    <Slider
                        value={inferenceOpacity}
                        onChange={onInferenceOpacityChange}
                        maxValue={1}
                        minValue={0}
                        step={0.01}
                    />
                </View>
            </DialogTrigger>
        </Flex>
    );
};
