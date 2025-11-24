// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { Item, Picker } from '@geti/ui';

import { useTrainedModels } from '../../../hooks/use-model';
import { useInference } from '../inference-provider.component';

export const ModelsPicker = () => {
    const models = useTrainedModels();
    const { selectedModelId, onSetSelectedModelId } = useInference();

    useEffect(() => {
        if (selectedModelId !== undefined || models.length === 0) {
            return;
        }

        onSetSelectedModelId(models[0].id);
    }, [selectedModelId, models, onSetSelectedModelId]);

    if (models === undefined || models.length === 0) {
        return null;
    }

    return (
        <Picker
            aria-label='Select model'
            items={models}
            selectedKey={selectedModelId}
            onSelectionChange={(key) => onSetSelectedModelId(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
