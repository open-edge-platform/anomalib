import { useState } from 'react';

import { Badge } from '@adobe/react-spectrum';
import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { ActiveIcon, LoadingIcon } from '@geti-inspect/icons';
import {
    ActionButton,
    AlertDialog,
    Cell,
    Column,
    DialogContainer,
    Flex,
    Heading,
    IllustratedMessage,
    Item,
    Menu,
    MenuTrigger,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    toast,
    View,
    type Key,
} from '@geti/ui';
import { Alert, Cancel, MoreMenu, Pending } from '@geti/ui/icons';
import { sortBy } from 'lodash-es';
import { useDateFormatter } from 'react-aria';
import { SchemaJob } from 'src/api/openapi-spec';

import { useProjectTrainingJobs, useRefreshModelsOnJobUpdates } from '../dataset/dataset-status-panel.component';
import { useInference } from '../inference-provider.component';
import { JobLogsDialog } from '../jobs/show-job-logs.component';
import { formatSize } from '../utils';

import classes from './models-view.module.scss';

const useModels = () => {
    const { projectId } = useProjectIdentifier();
    const modelsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
    const models = modelsQuery.data.models;

    return models;
};

interface ModelData {
    id: string;
    name: string;
    timestamp: string;
    startTime: number;
    durationInSeconds: number | null;
    status: 'Training' | 'Completed' | 'Failed';
    architecture: string;
    progress: number;
    job: SchemaJob | undefined;
    sizeBytes: number | null;
}

export const ModelsView = () => {
    const dateFormatter = useDateFormatter({ dateStyle: 'medium', timeStyle: 'short' });
    const { projectId } = useProjectIdentifier();

    const { jobs = [] } = useProjectTrainingJobs();
    useRefreshModelsOnJobUpdates(jobs);
    const [logsJobId, setLogsJobId] = useState<string | null>(null);
    const [modelPendingDelete, setModelPendingDelete] = useState<ModelData | null>(null);

    const models = useModels()
        .filter((model) => model.is_ready)
        .map((model): ModelData | null => {
            const job = jobs.find(({ id }) => id === model.train_job_id);
            if (job === undefined) {
                return null;
            }

            let timestamp = '';
            let durationInSeconds = 0;
            const start = job.start_time ? new Date(job.start_time) : new Date();
            if (job) {
                const end = job.end_time ? new Date(job.end_time) : new Date();
                durationInSeconds = Math.floor((end.getTime() - start.getTime()) / 1000);
                timestamp = dateFormatter.format(start);
            }

            return {
                id: model.id!,
                name: model.name!,
                status: 'Completed',
                architecture: model.name!,
                startTime: start.getTime(),
                timestamp,
                durationInSeconds,
                progress: 1.0,
                job,
                sizeBytes: model.size ?? null,
            };
        })
        .filter((model): model is ModelData => model !== null);

    const completedModelsJobsIDs = new Set(models.map((model) => model.job?.id));

    const nonCompletedJobs = jobs
        .filter((job) => !completedModelsJobsIDs.has(job.id))
        .map((job): ModelData => {
            const name = String(job.payload['model_name']);

            const start = job.start_time ? new Date(job.start_time) : new Date();
            const timestamp = dateFormatter.format(start);
            return {
                id: job.id!,
                name,
                status: job.status === 'pending' ? 'Training' : job.status === 'running' ? 'Training' : 'Failed',
                architecture: name,
                timestamp,
                startTime: start.getTime(),
                progress: job.progress ?? 0,
                durationInSeconds: null,
                job,
                sizeBytes: null,
            };
        });

    const showModels = sortBy([...nonCompletedJobs, ...models], (model) => -model.startTime);

    const { selectedModelId, onSetSelectedModelId } = useInference();

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

    const cancelJob = (jobId: string) => {
        cancelJobMutation.mutateAsync(
            {
                params: {
                    path: {
                        job_id: jobId,
                    },
                },
            },
            {
                onError: (_error) => {
                    toast({ type: 'error', message: 'Failed to cancel training job.' });
                },
            }
        );
    };

    const handleDeleteModel = async () => {
        if (modelPendingDelete === null) {
            return;
        }

        const targetModel = modelPendingDelete;

        try {
            await deleteModelMutation.mutateAsync({
                params: {
                    path: {
                        project_id: projectId,
                        model_id: targetModel.id,
                    },
                },
            });

            if (selectedModelId === targetModel.id) {
                onSetSelectedModelId(undefined);
            }

            toast({ type: 'success', message: `Model "${targetModel.name}" has been deleted.` });
        } catch (_error) {
            toast({ type: 'error', message: `Failed to delete "${targetModel.name}".` });
        } finally {
            setModelPendingDelete(null);
        }
    };

    return (
        <View backgroundColor='gray-100' height='100%'>
            {/* Models Table */}
            <TableView
                aria-label='Models'
                overflowMode='wrap'
                selectionStyle='highlight'
                selectionMode='single'
                selectedKeys={selectedModelId === undefined ? new Set() : new Set([selectedModelId])}
                onSelectionChange={(key) => {
                    if (typeof key === 'string') {
                        return;
                    }

                    const selectedId = key.values().next().value;
                    const selectedModel = models.find((model) => model.id === selectedId);

                    if (selectedModel?.status === 'Completed') {
                        onSetSelectedModelId(selectedModel?.id);
                    }
                }}
                UNSAFE_className={classes.table}
            >
                <TableHeader>
                    <Column width='2fr'>MODEL NAME</Column>
                    <Column align='end' width='1fr'>
                        MODEL SIZE
                    </Column>
                    <Column aria-label='Model actions' width='0fr'>
                        {' '}
                    </Column>
                </TableHeader>
                <TableBody>
                    {showModels.map((model) => (
                        <Row key={model.id}>
                            <Cell>
                                <Flex alignItems='start' gap='size-50' direction='column'>
                                    <Flex alignItems='end' gap='size-75'>
                                        <Text marginTop={'size-25'}>{model.name}</Text>
                                        {selectedModelId === model.id && (
                                            <Badge variant='info' UNSAFE_className={classes.badge}>
                                                <ActiveIcon />
                                                Active
                                            </Badge>
                                        )}
                                        {model.job?.status === 'pending' && (
                                            <Badge variant='neutral' UNSAFE_className={classes.badge}>
                                                <Pending />
                                                Pending
                                            </Badge>
                                        )}
                                        {model.job?.status === 'running' && (
                                            <Badge variant='info' UNSAFE_className={classes.badge}>
                                                <LoadingIcon />
                                                Training...
                                            </Badge>
                                        )}
                                        {model.job?.status === 'failed' && (
                                            <Badge variant='negative' UNSAFE_className={classes.badge}>
                                                <Alert />
                                                Failed
                                            </Badge>
                                        )}
                                        {model.job?.status === 'canceled' && (
                                            <Badge variant='neutral' UNSAFE_className={classes.badge}>
                                                <Cancel />
                                                Canceled
                                            </Badge>
                                        )}
                                    </Flex>
                                    <Text
                                        UNSAFE_style={{
                                            fontSize: '0.9rem',
                                            color: 'var(--spectrum-global-color-gray-500)',
                                        }}
                                    >
                                        {model.timestamp}
                                    </Text>
                                </Flex>
                            </Cell>
                            <Cell>
                                <Text>{formatSize(model.sizeBytes)}</Text>
                            </Cell>
                            <Cell>
                                <Flex justifyContent='end' alignItems='center'>
                                    <Flex alignItems='center' gap='size-200'>
                                        {(() => {
                                            const hasJobActions = Boolean(model.job?.id);
                                            const canDeleteModel =
                                                model.status === 'Completed' && model.id !== selectedModelId;
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

                                            return (
                                                <MenuTrigger>
                                                    <ActionButton isQuiet aria-label='model actions'>
                                                        <MoreMenu />
                                                    </ActionButton>
                                                    <Menu
                                                        disabledKeys={disabledMenuKeys}
                                                        onAction={(actionKey) => {
                                                            if (actionKey === 'logs' && model.job?.id) {
                                                                setLogsJobId(model.job.id);
                                                            }
                                                            if (actionKey === 'cancel' && model.job?.id) {
                                                                void cancelJob(model.job.id);
                                                            }
                                                            if (actionKey === 'delete' && canDeleteModel) {
                                                                setModelPendingDelete(model);
                                                            }
                                                        }}
                                                    >
                                                        {hasJobActions ? <Item key='logs'>View logs</Item> : null}
                                                        {model.job?.status === 'pending' ||
                                                        model.job?.status === 'running' ? (
                                                            <Item key='cancel'>Cancel training</Item>
                                                        ) : null}
                                                        {canDeleteModel ? <Item key='delete'>Delete model</Item> : null}
                                                    </Menu>
                                                </MenuTrigger>
                                            );
                                        })()}
                                    </Flex>
                                </Flex>
                            </Cell>
                        </Row>
                    ))}
                </TableBody>
            </TableView>

            {jobs.length === 0 && models.length === 0 && (
                <IllustratedMessage>
                    <Heading>No models in training</Heading>
                    <Text>Start a new training to see models here.</Text>
                </IllustratedMessage>
            )}

            <DialogContainer type='fullscreen' onDismiss={() => setLogsJobId(null)}>
                {logsJobId === null ? null : <JobLogsDialog close={() => setLogsJobId(null)} jobId={logsJobId} />}
            </DialogContainer>

            <DialogContainer onDismiss={() => setModelPendingDelete(null)}>
                {modelPendingDelete === null ? null : (
                    <AlertDialog
                        variant='destructive'
                        cancelLabel='Cancel'
                        title={`Delete model "${modelPendingDelete.name}"?`}
                        primaryActionLabel={deleteModelMutation.isPending ? 'Deleting...' : 'Delete model'}
                        isPrimaryActionDisabled={deleteModelMutation.isPending}
                        onPrimaryAction={handleDeleteModel}
                    >
                        Deleting a model removes any exported artifacts and cannot be undone.
                    </AlertDialog>
                )}
            </DialogContainer>
        </View>
    );
};
