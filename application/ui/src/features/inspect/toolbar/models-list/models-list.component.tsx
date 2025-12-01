// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Content, IllustratedMessage, Radio, RadioGroup, View } from '@geti/ui';
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

    const handleSelectionChange = (model_id: string) => {
        patchPipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id },
        });
    };

    if (isEmpty(modelsIds) || !selectedModelId) {
        return renderEmptyState();
    }

    return (
        <View padding={'size-250'} backgroundColor={'gray-200'} borderRadius={'regular'}>
            <RadioGroup
                isEmphasized
                defaultValue='dragon'
                aria-label='models list'
                value={selectedModelId}
                onChange={handleSelectionChange}
                isDisabled={patchPipeline.isPending}
            >
                {models.map((model) => (
                    <Radio key={model.id} value={String(model.id)}>
                        {model.name}
                    </Radio>
                ))}
            </RadioGroup>
        </View>
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
