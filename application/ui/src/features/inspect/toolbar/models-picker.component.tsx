// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { usePipeline, useSetModelToPipeline } from '@geti-inspect/hooks';
import { Item, Picker } from '@geti/ui';

import { useTrainedModels } from '../../../hooks/use-model';

export const ModelsPicker = () => {
    const models = useTrainedModels();
    const { data: pipeline } = usePipeline();
    const setModelToPipelineMutation = useSetModelToPipeline();

    const selectedModelId = pipeline.model?.id;

    useEffect(() => {
        if (selectedModelId !== undefined || models.length === 0) {
            return;
        }

        setModelToPipelineMutation(models[0].id);
    }, [selectedModelId, models, setModelToPipelineMutation]);

    if (models === undefined || models.length === 0) {
        return null;
    }

    return (
        <Picker
            aria-label='Select model'
            items={models}
            selectedKey={selectedModelId}
            onSelectionChange={(key) => setModelToPipelineMutation(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
