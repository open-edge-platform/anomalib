// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useEffect } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job } from '@geti-inspect/api/spec';
import { usePipeline, useProjectIdentifier, useSetModelToPipeline } from '@geti-inspect/hooks';
import { Flex, Loading, Text, View } from '@geti/ui';
import { WaitingIcon } from '@geti/ui/icons';

import { useTrainedModels } from '../../../hooks/use-model';
import { TrainingStatusItem } from './training-status-item.component';

const IdleItem = () => {
    const models = useTrainedModels();
    const { data: pipeline } = usePipeline();
    const setModelToPipelineMutation = useSetModelToPipeline();
    const selectedModelId = pipeline?.model?.id;

    useEffect(() => {
        if (selectedModelId !== undefined || models.length === 0) {
            return;
        }

        setModelToPipelineMutation(models[0].id);
    }, [selectedModelId, models, setModelToPipelineMutation]);

    return (
        <Flex
            alignItems='center'
            width='size-3000'
            justifyContent='start'
            gap='size-100'
            height='100%'
            UNSAFE_style={{
                padding: '0 var(--spectrum-global-dimension-size-200)',
            }}
        >
            <WaitingIcon height='14px' width='14px' stroke='var(--spectrum-global-color-gray-600)' />
            <Text marginStart={'5px'} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                Idle
            </Text>
        </Flex>
    );
};

const useCurrentJob = () => {
    const { data: jobsData } = $api.useSuspenseQuery('get', '/api/jobs', undefined, {
        refetchInterval: 5000,
    });

    const { projectId } = useProjectIdentifier();
    const runningJob = jobsData.jobs.find(
        (job: Job) => job.project_id === projectId && (job.status === 'running' || job.status === 'pending')
    );

    return runningJob;
};

export const ProgressBarItem = () => {
    const trainingJob = useCurrentJob();

    if (trainingJob !== undefined) {
        return <TrainingStatusItem trainingJob={trainingJob} />;
    }

    return <IdleItem />;
};

export const Footer = () => {
    return (
        <View gridArea={'footer'} backgroundColor={'gray-100'} width={'100%'} height={'size-400'} overflow={'hidden'}>
            <Suspense fallback={<Loading mode={'inline'} size='S' />}>
                <ProgressBarItem />
            </Suspense>
        </View>
    );
};
