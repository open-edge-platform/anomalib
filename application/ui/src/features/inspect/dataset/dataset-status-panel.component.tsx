import { Suspense, useEffect, useRef, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import {
    Button,
    Content,
    Divider,
    Flex,
    Heading,
    InlineAlert,
    IntelBrandedLoading,
    Item,
    Picker,
    ProgressBar,
    Text,
} from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { differenceBy, isEqual } from 'lodash-es';

import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface NotEnoughNormalImagesToTrainProps {
    mediaItemsCount: number;
}

const NotEnoughNormalImagesToTrain = ({ mediaItemsCount }: NotEnoughNormalImagesToTrainProps) => {
    const missingNormalImages = REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItemsCount;

    return (
        <InlineAlert variant='neutral'>
            <Heading>{missingNormalImages} images required</Heading>
            <Content>
                Capture {missingNormalImages} images of normal cases. They help the model learn what is standard, so it
                can better detect anomalies.
            </Content>
        </InlineAlert>
    );
};

const useAvailableModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/trainable-models', undefined, {
        staleTime: Infinity,
        gcTime: Infinity,
    });

    return data.trainable_models.map((model) => ({ id: model, name: model }));
};

const ReadyToTrain = () => {
    const startTrainingMutation = $api.useMutation('post', '/api/jobs:train');

    const availableModels = useAvailableModels();
    const { projectId } = useProjectIdentifier();
    const [selectedModel, setSelectedModel] = useState<string>(availableModels[0].id);

    const startTraining = () => {
        startTrainingMutation.mutate({
            body: { project_id: projectId, model_name: selectedModel },
        });
    };

    return (
        <InlineAlert variant='positive'>
            <Heading>Ready to train</Heading>
            <Content>
                <Flex direction={'column'} gap={'size-200'}>
                    <Text>You have enough normal images to train a model.</Text>

                    <Flex direction={'row'} alignItems={'end'} width={'100%'} gap={'size-200'} wrap={'wrap'}>
                        <Picker
                            label={'Model'}
                            selectedKey={selectedModel}
                            onSelectionChange={(key) => key !== null && setSelectedModel(String(key))}
                        >
                            {availableModels.map((model) => (
                                <Item key={model.id}>{model.name}</Item>
                            ))}
                        </Picker>

                        <Button isPending={startTrainingMutation.isPending} onPress={startTraining}>
                            Start training
                        </Button>
                    </Flex>
                </Flex>
            </Content>
        </InlineAlert>
    );
};

interface TrainingInProgressProps {
    job: Job;
}

const TrainingInProgress = ({ job }: TrainingInProgressProps) => {
    if (job === undefined) {
        return null;
    }

    if (job.status === 'pending') {
        const heading = `Training will start soon - ${job.payload.model_name}`;

        return (
            <InlineAlert variant='info'>
                <Heading>{heading}</Heading>
                <Content>
                    <Flex direction={'column'} gap={'size-100'}>
                        <Text>{job.message}</Text>
                        <ProgressBar aria-label='Training progress' isIndeterminate />
                    </Flex>
                </Content>
            </InlineAlert>
        );
    }

    if (job.status === 'running') {
        const heading = `Training in progress - ${job.payload.model_name}`;

        return (
            <InlineAlert variant='info'>
                <Heading>{heading}</Heading>
                <Content>
                    <Flex direction={'column'} gap={'size-100'}>
                        <Text>{job.message}</Text>
                        <ProgressBar value={job.progress} aria-label='Training progress' />
                    </Flex>
                </Content>
            </InlineAlert>
        );
    }

    if (job.status === 'failed') {
        const heading = `Training failed - ${job.payload.model_name}`;

        return (
            <InlineAlert variant='negative'>
                <Heading>{heading}</Heading>
                <Content>
                    <Text>{job.message}</Text>
                </Content>
            </InlineAlert>
        );
    }

    if (job.status === 'canceled') {
        const heading = `Training canceled - ${job.payload.model_name}`;

        return (
            <InlineAlert variant='negative'>
                <Heading>{heading}</Heading>
                <Content>
                    <Text>{job.message}</Text>
                </Content>
            </InlineAlert>
        );
    }

    if (job.status === 'completed') {
        const heading = `Training completed - ${job.payload.model_name}`;

        return (
            <InlineAlert variant='positive'>
                <Heading>{heading}</Heading>
                <Content>
                    <Text>{job.message}</Text>
                </Content>
            </InlineAlert>
        );
    }

    return null;
};

const REFETCH_INTERVAL_WITH_TRAINING = 1_000;

const useProjectTrainingJobs = () => {
    const queryClient = useQueryClient();
    const { projectId } = useProjectIdentifier();
    const prevJobsRef = useRef<Job[]>([]);

    const { data } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: ({ state }) => {
            const projectHasTrainingJob = state.data?.jobs.some(
                ({ project_id, type, status }) =>
                    projectId === project_id && type === 'training' && (status === 'running' || status === 'pending')
            );

            return projectHasTrainingJob ? REFETCH_INTERVAL_WITH_TRAINING : undefined;
        },
    });

    useEffect(() => {
        if (data === undefined) {
            return;
        }

        if (!isEqual(prevJobsRef.current, data?.jobs)) {
            const differenceInJobsBasedOnStatus = differenceBy(prevJobsRef.current, data.jobs, (job) => job.status);
            const shouldRefetchModels = differenceInJobsBasedOnStatus.some((job) => job.status === 'completed');

            if (shouldRefetchModels) {
                queryClient.invalidateQueries({
                    queryKey: [
                        'get',
                        '/api/projects/{project_id}/models',
                        { params: { path: { project_id: projectId } } },
                    ],
                });
            }
        }

        prevJobsRef.current = data?.jobs ?? [];
    }, [data, queryClient, projectId]);

    return { jobs: data?.jobs.filter((job) => job.project_id === projectId) };
};

const TrainingInProgressList = () => {
    const { jobs } = useProjectTrainingJobs();

    if (jobs === undefined || jobs.length === 0) {
        return null;
    }

    return (
        <>
            <Flex direction={'column'} gap={'size-50'} height={'size-2000'} UNSAFE_style={{ overflowY: 'auto' }}>
                {jobs?.map((job) => <TrainingInProgress job={job} key={job.id} />)}
            </Flex>
            <Divider size={'S'} />
        </>
    );
};

interface DatasetStatusPanelProps {
    mediaItemsCount: number;
}

export const DatasetStatusPanel = ({ mediaItemsCount }: DatasetStatusPanelProps) => {
    if (mediaItemsCount < REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING) {
        return <NotEnoughNormalImagesToTrain mediaItemsCount={mediaItemsCount} />;
    }

    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <TrainingInProgressList />
            <ReadyToTrain />
        </Suspense>
    );
};
