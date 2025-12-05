// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Content, Flex, IllustratedMessage } from '@geti/ui';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';
import { NotFound } from 'packages/ui/icons';

import { useTrainedModels } from '../../../../hooks/use-trained-models';

import classes from './model-list.module.scss';

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

    if (isEmpty(modelsIds)) {
        return renderEmptyState();
    }

    return (
        <Flex direction='column' gap='size-100'>
            {models.map((model) => (
                <Button
                    key={model.id}
                    variant='secondary'
                    onPress={() => handleSelectionChange(String(model.id))}
                    height={'size-800'}
                    isPending={patchPipeline.isPending}
                    isDisabled={patchPipeline.isPending}
                    UNSAFE_className={clsx(classes.option, { [classes.active]: model.id === selectedModelId })}
                >
                    {model.name}
                </Button>
            ))}
        </Flex>
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
