import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { usePatchPipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { ActionButton, AlertDialog, DialogContainer, Item, Menu, MenuTrigger, toast, type Key } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { JobLogsDialog } from '../jobs/show-job-logs.component';
import { ExportModelDialog } from './export-model-dialog.component';
import type { ModelData } from './model-types';

interface ModelActionsMenuProps {
    model: ModelData;
    selectedModelId: string | undefined;
}

type DialogType = 'logs' | 'delete' | 'export' | null;

export const ModelActionsMenu = ({ model, selectedModelId }: ModelActionsMenuProps) => {
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const [openDialog, setOpenDialog] = useState<DialogType>(null);

    const cancelJobMutation = $api.useMutation('post', '/api/jobs/{job_id}:cancel');
    const deleteModelMutation = $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
                ['get', '/api/jobs'],
            ],
        },
    });

    const hasJobActions = Boolean(model.job?.id);
    const hasCompletedStatus = model.status === 'Completed';
    const canDeleteModel = hasCompletedStatus && model.id !== selectedModelId;
    const canExportModel = hasCompletedStatus;
    const shouldShowMenu = hasJobActions || canDeleteModel || canExportModel;

    if (!shouldShowMenu) {
        return null;
    }

    const disabledMenuKeys: Key[] = [];
    if (cancelJobMutation.isPending) {
        disabledMenuKeys.push('cancel');
    }
    if (deleteModelMutation.isPending) {
        disabledMenuKeys.push('delete');
    }
    if (model.id === selectedModelId || patchPipeline.isPending) {
        disabledMenuKeys.push('activate');
    }

    const handleCancelJob = () => {
        if (!model.job?.id) {
            return;
        }

        void cancelJobMutation.mutateAsync(
            { params: { path: { job_id: model.job.id } } },
            {
                onError: () => {
                    toast({ type: 'error', message: 'Failed to cancel training job.' });
                },
            }
        );
    };

    const handleSetModel = (modelId?: string) => {
        patchPipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { model_id: modelId } });
    };

    const handleDeleteModel = () => {
        void deleteModelMutation.mutateAsync(
            { params: { path: { project_id: projectId, model_id: model.id } } },
            {
                onSuccess: () => {
                    if (selectedModelId === model.id) {
                        handleSetModel(undefined);
                    }

                    toast({ type: 'success', message: `Model "${model.name}" has been deleted.` });
                },
                onError: () => {
                    toast({ type: 'error', message: `Failed to delete "${model.name}".` });
                },
                onSettled: () => {
                    setOpenDialog(null);
                },
            }
        );
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label='model actions'>
                    <MoreMenu />
                </ActionButton>
                <Menu
                    disabledKeys={disabledMenuKeys}
                    onAction={(actionKey) => {
                        if (actionKey === 'logs' && model.job?.id) {
                            setOpenDialog('logs');
                        }
                        if (actionKey === 'cancel' && model.job?.id) {
                            void handleCancelJob();
                        }
                        if (actionKey === 'export' && canExportModel) {
                            setOpenDialog('export');
                        }
                        if (actionKey === 'delete' && canDeleteModel) {
                            setOpenDialog('delete');
                        }
                        if (actionKey === 'activate' && hasCompletedStatus) {
                            handleSetModel(model.id);
                        }
                    }}
                >
                    {hasCompletedStatus ? <Item key='activate'>Activate</Item> : null}
                    {hasJobActions ? <Item key='logs'>View logs</Item> : null}
                    {model.job?.status === 'pending' || model.job?.status === 'running' ? (
                        <Item key='cancel'>Cancel training</Item>
                    ) : null}
                    {canExportModel ? <Item key='export'>Export model</Item> : null}
                    {canDeleteModel ? <Item key='delete'>Delete model</Item> : null}
                </Menu>
            </MenuTrigger>

            <DialogContainer type='fullscreen' onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'logs' && model.job?.id ? (
                    <JobLogsDialog close={() => setOpenDialog(null)} jobId={model.job.id} />
                ) : null}
            </DialogContainer>

            <DialogContainer onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'delete' && canDeleteModel ? (
                    <AlertDialog
                        variant='destructive'
                        cancelLabel='Cancel'
                        title={`Delete model "${model.name}"?`}
                        primaryActionLabel={deleteModelMutation.isPending ? 'Deleting...' : 'Delete model'}
                        isPrimaryActionDisabled={deleteModelMutation.isPending}
                        onPrimaryAction={() => {
                            void handleDeleteModel();
                        }}
                    >
                        Deleting a model removes any exported artifacts and cannot be undone.
                    </AlertDialog>
                ) : null}
            </DialogContainer>

            <DialogContainer onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'export' && canExportModel ? (
                    <ExportModelDialog model={model} close={() => setOpenDialog(null)} />
                ) : null}
            </DialogContainer>
        </>
    );
};
