// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Item, Picker } from '@geti/ui';

import { useInference } from '../inference-provider.component';

const useTrainedModels = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data?.models.map((model) => ({ id: model.id, name: model.name })) || [];
};

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
