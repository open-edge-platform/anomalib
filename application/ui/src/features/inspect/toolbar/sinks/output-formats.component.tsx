// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, Flex } from '@geti/ui';

import { OutputFormat, SinkOutputFormats } from './utils';

type OutputFormatsProps = {
    config?: SinkOutputFormats;
};

export const OutputFormats = ({ config = [] }: OutputFormatsProps) => {
    return (
        <Flex wrap='wrap' gap='size-100'>
            <Checkbox
                name='output_formats'
                value={OutputFormat.PREDICTIONS}
                defaultSelected={config?.includes(OutputFormat.PREDICTIONS)}
            >
                Predictions
            </Checkbox>
            <Checkbox
                name='output_formats'
                value={OutputFormat.IMAGE_ORIGINAL}
                defaultSelected={config?.includes(OutputFormat.IMAGE_ORIGINAL)}
            >
                Image Original
            </Checkbox>
            <Checkbox
                name='output_formats'
                value={OutputFormat.IMAGE_WITH_PREDICTIONS}
                defaultSelected={config?.includes(OutputFormat.IMAGE_WITH_PREDICTIONS)}
            >
                Image with Predictions
            </Checkbox>
        </Flex>
    );
};
