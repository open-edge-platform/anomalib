import { Badge } from '@adobe/react-spectrum';
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
import { SchemaJob } from 'src/api/openapi-spec';

import { useProjectTrainingJobs, useRefreshModelsOnJobUpdates } from '../dataset/dataset-status-panel.component';
import { useInference } from '../inference-provider.component';
import { ShowJobLogs } from '../jobs/show-job-logs.component';

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
}

export const ModelsView = () => {
    const dateFormatter = useDateFormatter({ dateStyle: 'medium', timeStyle: 'short' });

    const { jobs = [] } = useProjectTrainingJobs();
    useRefreshModelsOnJobUpdates(jobs);

    const models = useModels()
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
            };
        })
        .filter((model): model is ModelData => model !== null);

    const nonCompletedJobs = jobs
        .filter((job) => job.status !== 'completed')
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
                progress: 1.0,
                durationInSeconds: null,
                job,
            };
        });

    const showModels = sortBy([...nonCompletedJobs, ...models], (model) => -model.startTime);

    const { selectedModelId, onSetSelectedModelId } = useInference();

    return (
        <View backgroundColor='gray-100' height='100%'>
            <View borderTopWidth='thin' borderTopColor='gray-400' backgroundColor={'gray-300'}>
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

                        onSetSelectedModelId(selectedModel?.id);
                    }}
                >
                    <TableHeader>
                        <Column>MODEL NAME</Column>
                        <Column> </Column>
                    </TableHeader>
                    <TableBody>
                        {showModels.map((model) => (
                            <Row key={model.id}>
                                <Cell>
                                    <Flex alignItems='start' gap='size-25' direction='column'>
                                        <Text>{model.name}</Text>
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
                                    <Flex justifyContent='end' alignItems='center'>
                                        <Flex alignItems='center' gap='size-200'>
                                            {model.job?.status === 'pending' && <span>pending...</span>}
                                            {model.job?.status === 'running' && <span>{model.job.progress}%...</span>}
                                            {model.job?.status === 'canceled' && (
                                                <Badge variant='neutral'>Cancelled</Badge>
                                            )}
                                            {model.job?.status === 'failed' && <Badge variant='negative'>Failed</Badge>}
                                            {selectedModelId === model.id && <Badge variant='info'>Active</Badge>}
                                            {model.job?.id && <ShowJobLogs jobId={model.job.id} />}
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
            </View>
        </View>
    );
};
