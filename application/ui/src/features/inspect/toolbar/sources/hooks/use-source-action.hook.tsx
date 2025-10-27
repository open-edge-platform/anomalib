// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { toast } from '@geti/ui';
import { usePatchPipeline } from 'src/hooks/use-pipeline.hook';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

import { SourceConfig } from '../util';
import { useSourceMutation } from './use-source-mutation.hook';

interface useSourceActionProps<T> {
    config: Awaited<T>;
    isNewSource: boolean;
    bodyFormatter: (formData: FormData) => T;
}

export const useSourceAction = <T extends SourceConfig>({
    config,
    isNewSource,
    bodyFormatter,
}: useSourceActionProps<T>) => {
    const pipeline = usePatchPipeline();
    const { projectId } = useProjectIdentifier();
    const addOrUpdateSource = useSourceMutation(isNewSource);

    return useActionState<T, FormData>(async (_prevState: T, formData: FormData) => {
        const body = bodyFormatter(formData);

        try {
            const source_id = await addOrUpdateSource(body);

            await pipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { source_id },
            });

            toast({
                type: 'success',
                message: `Source configuration ${isNewSource ? 'created' : 'updated'} successfully.`,
            });

            return { ...body, id: source_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save source configuration, ${details ?? 'please try again'}`,
            });
        }

        return body;
    }, config);
};
