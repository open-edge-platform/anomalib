import { ComponentProps, Suspense, useEffect, useRef } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job, SchemaJob, SchemaJobStatus } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Content, Flex, Heading, InlineAlert, IntelBrandedLoading, ProgressBar, Text } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { isEqual } from 'lodash-es';

import { ShowJobLogs } from '../jobs/show-job-logs.component';
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

interface TrainingInProgressProps {
    job: Job;
}

const statusToVariant: Record<SchemaJobStatus, ComponentProps<typeof InlineAlert>['variant']> = {
    pending: 'info',
    running: 'info',
    completed: 'positive',
    canceled: 'negative',
    failed: 'negative',
};

function getHeading(job: SchemaJob) {
    if (job.status === 'pending') {
        return `Training will start soon - ${job.payload.model_name}`;
    }
    if (job.status === 'running') {
        return `Training in progress - ${job.payload.model_name}`;
    }

    if (job.status === 'failed') {
        return `Training failed - ${job.payload.model_name}`;
    }

    if (job.status === 'canceled') {
        return `Training canceled - ${job.payload.model_name}`;
    }

    if (job.status === 'completed') {
        return `Training completed - ${job.payload.model_name}`;
    }
    return null;
}

const TrainingInProgress = ({ job }: TrainingInProgressProps) => {
    if (job === undefined) {
        return null;
    }

    const variant = statusToVariant[job.status];
    const heading = getHeading(job);

    return (
        <InlineAlert variant={variant}>
            <Heading>
                <Flex gap='size-100' alignItems={'center'} justifyContent={'space-between'}>
                    {heading}
                    {job.id && <ShowJobLogs jobId={job.id} />}
                </Flex>
            </Heading>
            <Content>
                <Flex direction={'column'} gap={'size-100'}>
                    <Text>{job.message}</Text>
                    {job.status === 'pending' && <ProgressBar aria-label='Training progress' isIndeterminate />}
                </Flex>
            </Content>
        </InlineAlert>
    );
};

const REFETCH_INTERVAL_WITH_TRAINING = 1_000;

export const useProjectTrainingJobs = () => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: ({ state }) => {
            const projectHasTrainingJob = state.data?.jobs.some(
                ({ project_id, type, status }) =>
                    projectId === project_id && type === 'training' && (status === 'running' || status === 'pending')
            );

            return projectHasTrainingJob ? REFETCH_INTERVAL_WITH_TRAINING : undefined;
        },
    });

    return { jobs: data?.jobs.filter((job) => job.project_id === projectId) };
};

export const useRefreshModelsOnJobUpdates = (jobs: Job[] | undefined) => {
    const queryClient = useQueryClient();
    const { projectId } = useProjectIdentifier();
    const prevJobsRef = useRef<Job[]>([]);

    useEffect(() => {
        if (jobs === undefined) {
            return;
        }

        if (!isEqual(prevJobsRef.current, jobs)) {
            const shouldRefetchModels = jobs.some((job, idx) => {
                // NOTE: assuming index stays the same
                return job.status === 'completed' && job.status !== prevJobsRef.current.at(idx)?.status;
            });

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

        prevJobsRef.current = jobs ?? [];
    }, [jobs, queryClient, projectId]);
};

const TrainingInProgressList = () => {
    const { jobs } = useProjectTrainingJobs();
    useRefreshModelsOnJobUpdates(jobs);

    if (jobs === undefined || jobs.length === 0) {
        return null;
    }

    return (
        <Flex direction={'column'} gap={'size-50'} UNSAFE_style={{ overflowY: 'auto' }}>
            {jobs?.map((job) => <TrainingInProgress job={job} key={job.id} />)}
        </Flex>
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
        </Suspense>
    );
};
