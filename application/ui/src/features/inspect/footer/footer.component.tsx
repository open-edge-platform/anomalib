// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useEffect } from 'react';

import { usePipeline, useSetModelToPipeline } from '@geti-inspect/hooks';
import { Loading, View } from '@geti/ui';

import { useTrainedModels } from '../../../hooks/use-model';
import { ConnectionStatusAdapter, TrainingStatusAdapter } from './adapters';
import { StatusBar } from './status-bar';

const AutoSelectModel = () => {
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

    return null;
};

export const Footer = () => {
    return (
        <View gridArea={'footer'} backgroundColor={'gray-100'} width={'100%'} height={'size-400'} overflow={'hidden'}>
            <AutoSelectModel />
            <Suspense fallback={<Loading mode={'inline'} size='S' />}>
                <ConnectionStatusAdapter />
                <TrainingStatusAdapter />
                <StatusBar />
            </Suspense>
        </View>
    );
};
