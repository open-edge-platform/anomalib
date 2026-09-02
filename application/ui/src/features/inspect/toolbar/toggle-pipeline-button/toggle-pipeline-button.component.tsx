// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActivatePipeline, useDisablePipeline, usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { Switch } from 'packages/ui';
import { toast } from 'src/components/toast/toast.component';

export const TogglePipelineButton = () => {
    const pipelineQuery = usePipeline();
    const { projectId } = useProjectIdentifier();
    const enablePipelineMutation = useActivatePipeline({});
    const disablePipelineMutation = useDisablePipeline(projectId);

    const isPending = disablePipelineMutation.isPending || enablePipelineMutation.isPending;
    const isPipelineEnabled = pipelineQuery.data?.status === 'running';

    const handleToggle = () => {
        const mutationOptions = {
            onSuccess: () => {
                toast({
                    type: 'success',
                    message: `Pipeline ${isPipelineEnabled ? 'disabled' : 'enabled'} successfully`,
                });
            },
        };

        if (isPipelineEnabled) {
            disablePipelineMutation.mutate({ params: { path: { project_id: projectId } } }, mutationOptions);
        } else {
            enablePipelineMutation.mutate({ params: { path: { project_id: projectId } } }, mutationOptions);
        }
    };

    return (
        <Switch
            isEmphasized
            isSelected={isPipelineEnabled}
            isDisabled={isPending}
            onChange={handleToggle}
            UNSAFE_style={{ margin: '0px' }}
        >
            Pipeline {isPipelineEnabled ? 'enabled' : 'disabled'}
        </Switch>
    );
};
