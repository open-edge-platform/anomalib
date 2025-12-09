// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Content, Flex, IllustratedMessage, Loading } from '@geti/ui';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';
import { NotFound } from 'packages/ui/icons';

import { useGetModels } from '../../../..//hooks/use-get-models.hook';
import { useListEnd } from '../../../../hooks/use-list-end.hook';

import classes from './model-list.module.scss';

export const ModelsList = () => {
    const { models, isLoading, isFetchingNextPage, fetchNextPage } = useGetModels();
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const { data: pipeline } = usePipeline();
    const sentinelRef = useListEnd({ onEndReached: fetchNextPage, disabled: isLoading });

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
        <Flex direction='column' gap='size-100' maxHeight={'60vh'} ref={sentinelRef}>
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

            {isFetchingNextPage && <Loading mode='inline' size='S' />}
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
