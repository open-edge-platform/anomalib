import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import {
    Cell,
    Column,
    Flex,
    Heading,
    IllustratedMessage,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    View,
} from '@geti/ui';
import { sortBy } from 'lodash-es';
import { useDateFormatter } from 'react-aria';

import { useProjectTrainingJobs, useRefreshModelsOnJobUpdates } from '../dataset/dataset-status-panel.component';
import { useInference } from '../inference-provider.component';
import { formatSize } from '../utils';
import { ModelActionsMenu } from './model-actions-menu.component';
import { ModelStatusBadges } from './model-status-badges.component';
import { ModelData } from './model-types';

import classes from './models-view.module.scss';

const useModels = () => {
    const { projectId } = useProjectIdentifier();
    const modelsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
    const models = modelsQuery.data.models;

    return models;
};

export const ModelsView = () => {
    const dateFormatter = useDateFormatter({ dateStyle: 'medium', timeStyle: 'short' });

    const { jobs = [] } = useProjectTrainingJobs();
    useRefreshModelsOnJobUpdates(jobs);

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
                renderEmptyState={() => (
                    <IllustratedMessage>
                        <Heading>No models in training</Heading>
                        <Text>Start a new training to see models here.</Text>
                    </IllustratedMessage>
                )}
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
                                        <Text marginTop={'size-25'} UNSAFE_className={classes.modelName}>
                                            {model.name}
                                        </Text>
                                        <ModelStatusBadges
                                            isSelected={selectedModelId === model.id}
                                            jobStatus={model.job?.status}
                                        />
                                    </Flex>
                                    <Text UNSAFE_className={classes.modelTimestamp}>{model.timestamp}</Text>
                                </Flex>
                            </Cell>
                            <Cell>
                                <Text UNSAFE_className={classes.modelSize}>{formatSize(model.sizeBytes)}</Text>
                            </Cell>
                            <Cell>
                                <Flex justifyContent='end' alignItems='center'>
                                    <Flex alignItems='center' gap='size-200'>
                                        <ModelActionsMenu
                                            model={model}
                                            selectedModelId={selectedModelId}
                                            onSetSelectedModelId={onSetSelectedModelId}
                                        />
                                    </Flex>
                                </Flex>
                            </Cell>
                        </Row>
                    ))}
                </TableBody>
            </TableView>
        </View>
    );
};
