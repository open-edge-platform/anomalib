// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, Text } from '@geti/ui';
import { Gauge1, Gauge3, Gauge5 } from '@geti/ui/icons';
import { isNil } from 'lodash-es';

import type { PerformanceMetrics } from './types';

import classes from './train-model.module.scss';

const GaugeIcons = {
    1: Gauge1,
    2: Gauge3,
    3: Gauge5,
} as const;

interface MetricRatingBarProps {
    label: string;
    value?: number | null;
}

const MetricRatingBar = ({ label, value }: MetricRatingBarProps) => {
    if (isNil(value)) return null;

    const GaugeIcon = GaugeIcons[value as 1 | 2 | 3];
    if (!GaugeIcon) return null;

    return (
        <Flex direction='column' alignItems='center' gap='size-100'>
            <Heading margin={0} UNSAFE_className={classes.attributeRatingTitle}>
                {label}
            </Heading>
            <GaugeIcon size='M' UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-800)' }} />
        </Flex>
    );
};

interface ModelMetricsDisplayProps {
    metrics?: PerformanceMetrics;
    parameters?: number | null;
}

export const ModelMetricsDisplay = ({ metrics, parameters }: ModelMetricsDisplayProps) => {
    const hasMetrics = metrics?.training || metrics?.inference;
    const hasParams = !isNil(parameters);

    if (!hasMetrics && !hasParams) return null;

    return (
        <Flex gap='size-400'>
            <MetricRatingBar label='Training speed' value={metrics?.training} />
            <MetricRatingBar label='Inference speed' value={metrics?.inference} />
            {hasParams && (
                <Flex direction='column' alignItems='center' gap='size-100'>
                    <Heading margin={0} UNSAFE_className={classes.attributeRatingTitle}>
                        Params
                    </Heading>
                    <Text UNSAFE_className={classes.attributeRatingValue}>{parameters}M</Text>
                </Flex>
            )}
        </Flex>
    );
};
