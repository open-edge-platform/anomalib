// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Content, IllustratedMessage, Item, ListView, Selection } from '@geti/ui';
import { isEmpty } from 'lodash-es';
import { NotFound } from 'packages/ui/icons';

import { useTrainedModels } from '../../../../hooks/use-model';

export const ModelsList = () => {
    const models = useTrainedModels();
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const { data: pipeline } = usePipeline();

    const selectedModelId = pipeline.model?.id;
    const modelsIds = models.map((model) => model.id).filter(Boolean) as string[];

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;

        patchPipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id: String(firstKey) },
        });
    };

    return (
        <ListView
            items={models}
            density='spacious'
            maxWidth='100%'
            height={isEmpty(modelsIds) ? 'size-3000' : undefined}
            maxHeight='size-5000'
            selectionMode='single'
            disabledBehavior='all'
            aria-label='models list'
            renderEmptyState={renderEmptyState}
            onSelectionChange={handleSelectionChange}
            selectedKeys={isEmpty(models) ? new Set() : new Set([String(selectedModelId)])}
            loadingState={patchPipeline.isPending && !isEmpty(models) ? 'loading' : 'idle'}
            disabledKeys={patchPipeline.isPending ? modelsIds : []}
        >
            {(item) => <Item>{item.name}</Item>}
        </ListView>
    );
};

function renderEmptyState() {
    return (
        <IllustratedMessage>
            <NotFound />
            <Content>No models trained yet</Content>
        </IllustratedMessage>
    );
}
