import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { ActionButton, AlertDialog, DialogContainer, Item, Menu, MenuTrigger, toast, type Key } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { JobLogsDialog } from '../jobs/show-job-logs.component';
import type { ModelData } from './model-types';

interface ModelActionsMenuProps {
    model: ModelData;
    selectedModelId: string | undefined;
    onSetSelectedModelId: (modelId: string | undefined) => void;
}

export const ModelActionsMenu = ({ model, selectedModelId, onSetSelectedModelId }: ModelActionsMenuProps) => {
    const { projectId } = useProjectIdentifier();
    const [isLogsDialogOpen, setIsLogsDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

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
    const canDeleteModel = model.status === 'Completed' && model.id !== selectedModelId;
    const shouldShowMenu = hasJobActions || canDeleteModel;

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

    const handleCancelJob = () => {
        if (!model.job?.id) {
            return;
        }

        void cancelJobMutation.mutateAsync(
            {
                params: {
                    path: {
                        job_id: model.job.id,
                    },
                },
            },
            {
                onError: () => {
                    toast({ type: 'error', message: 'Failed to cancel training job.' });
                },
            }
        );
    };

    const handleDeleteModel = () => {
        void deleteModelMutation.mutateAsync(
            {
                params: {
                    path: {
                        project_id: projectId,
                        model_id: model.id,
                    },
                },
            },
            {
                onSuccess: () => {
                    if (selectedModelId === model.id) {
                        onSetSelectedModelId(undefined);
                    }

                    toast({ type: 'success', message: `Model "${model.name}" has been deleted.` });
                },
                onError: () => {
                    toast({ type: 'error', message: `Failed to delete "${model.name}".` });
                },
                onSettled: () => {
                    setIsDeleteDialogOpen(false);
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
                            setIsLogsDialogOpen(true);
                        }
                        if (actionKey === 'cancel' && model.job?.id) {
                            void handleCancelJob();
                        }
                        if (actionKey === 'delete' && canDeleteModel) {
                            setIsDeleteDialogOpen(true);
                        }
                    }}
                >
                    {hasJobActions ? <Item key='logs'>View logs</Item> : null}
                    {model.job?.status === 'pending' || model.job?.status === 'running' ? (
                        <Item key='cancel'>Cancel training</Item>
                    ) : null}
                    {canDeleteModel ? <Item key='delete'>Delete model</Item> : null}
                </Menu>
            </MenuTrigger>

            <DialogContainer type='fullscreen' onDismiss={() => setIsLogsDialogOpen(false)}>
                {isLogsDialogOpen && model.job?.id ? (
                    <JobLogsDialog close={() => setIsLogsDialogOpen(false)} jobId={model.job.id} />
                ) : null}
            </DialogContainer>

            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {!isDeleteDialogOpen || !canDeleteModel ? null : (
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
                )}
            </DialogContainer>
        </>
    );
};
