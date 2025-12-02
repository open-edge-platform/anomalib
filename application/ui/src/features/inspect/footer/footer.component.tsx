// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useEffect } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job } from '@geti-inspect/api/spec';
import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, Loading, Text, View } from '@geti/ui';
import { WaitingIcon } from '@geti/ui/icons';

import { useTrainedModels } from '../../../hooks/use-model';
import { TrainingStatusItem } from './training-status-item.component';

const useCurrentJob = () => {
    const { projectId } = useProjectIdentifier();
    const { data: jobsData } = $api.useSuspenseQuery('get', '/api/jobs', undefined, {
        refetchInterval: 5000,
    });

    const runningJob = jobsData.jobs.find(
        (job: Job) => job.project_id === projectId && (job.status === 'running' || job.status === 'pending')
    );

    return runningJob;
};

const useDefaultModel = () => {
    const models = useTrainedModels();
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);

    const hasSelectedModel = pipeline?.model?.id !== undefined;
    const hasNonAvailableModels = models.length === 0;

    useEffect(() => {
        if (hasSelectedModel || hasNonAvailableModels || patchPipeline.isPending) {
            return;
        }

        patchPipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id: models[0].id },
        });
    }, [hasNonAvailableModels, hasSelectedModel, models, patchPipeline, projectId]);
};

export const ProgressBarItem = () => {
    const trainingJob = useCurrentJob();

    if (trainingJob !== undefined) {
        return <TrainingStatusItem trainingJob={trainingJob} />;
    }

    return (
        <Flex
            gap='size-100'
            height='100%'
            width='size-3000'
            alignItems='center'
            justifyContent='start'
            UNSAFE_style={{ padding: '0 var(--spectrum-global-dimension-size-200)' }}
        >
            <WaitingIcon height='14px' width='14px' stroke='var(--spectrum-global-color-gray-600)' />
            <Text marginStart={'5px'} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                Idle
            </Text>
        </Flex>
    );
};

export const Footer = () => {
    useDefaultModel();

    return (
        <View gridArea={'footer'} backgroundColor={'gray-100'} width={'100%'} height={'size-400'} overflow={'hidden'}>
            <Suspense fallback={<Loading mode={'inline'} size='S' />}>
                <ProgressBarItem />
            </Suspense>
        </View>
    );
};
