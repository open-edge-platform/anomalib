// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job, SchemaJob } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, ProgressBar, Text, View } from '@geti/ui';
import { CanceledIcon, WaitingIcon } from '@geti/ui/icons';
import { queryOptions, experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';
import { fetchSSE } from 'src/api/fetch-sse';

const IdleItem = () => {
    return (
        <Flex
            alignItems='center'
            width='size-3000'
            justifyContent='start'
            gap='size-100'
            height='100%'
            UNSAFE_style={{
                padding: '0 var(--spectrum-global-dimension-size-200)',
            }}
        >
            <WaitingIcon height='14px' width='14px' stroke='var(--spectrum-global-color-gray-600)' />
            <Text marginStart={'5px'} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                Idle
            </Text>
        </Flex>
    );
};

const getStyleForMessage = (message: string) => {
    if (message.toLowerCase().includes('valid')) {
        return {
            backgroundColor: 'var(--spectrum-global-color-yellow-600)',
            color: '#000',
        };
    } else if (message.toLowerCase().includes('test')) {
        return {
            backgroundColor: 'var(--spectrum-global-color-green-600)',
            color: '#fff',
        };
    } else if (message.toLowerCase().includes('train') || message.toLowerCase().includes('fit')) {
        return {
            backgroundColor: 'var(--spectrum-global-color-blue-600)',
            color: '#fff',
        };
    }

    return {
        backgroundColor: 'var(--spectrum-global-color-blue-600)',
        color: '#fff',
    };
};

const TrainingStatusItem = ({ trainingJob }: { trainingJob: SchemaJob }) => {
    // Cancel training job
    const cancelJobMutation = $api.useMutation('post', '/api/jobs/{job_id}:cancel');
    const handleCancel = async () => {
        try {
            if (trainingJob.id === undefined) {
                throw Error('TODO: jobs should always have an ID');
            }

            console.info('Cancel training');
            await cancelJobMutation.mutateAsync({
                params: {
                    path: {
                        job_id: trainingJob.id,
                    },
                },
            });
            console.info('Job cancelled successfully');
        } catch (error) {
            console.error('Failed to cancel job:', error);
        }
    };

    const progressQuery = useQuery(
        queryOptions({
            queryKey: ['get', '/api/jobs/{job_id}/progress', trainingJob.id],
            queryFn: streamedQuery({
                queryFn: () => fetchSSE(`/api/jobs/${trainingJob.id}/progress`),
                maxChunks: 1,
            }),
            staleTime: Infinity,
        })
    );

    // Get the job progress and message from the last SSE message, or fallback
    const lastJobProgress = progressQuery.data?.at(-1);
    const progress = lastJobProgress?.progress ?? trainingJob.progress;
    const message = lastJobProgress?.message ?? trainingJob.message;

    const { backgroundColor, color } = getStyleForMessage(message);

    return (
        <div
            style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor,
            }}
        >
            <Flex direction='row' alignItems='center' width='100px' justifyContent='space-between'>
                <button
                    onClick={() => {
                        handleCancel();
                    }}
                    style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                    }}
                >
                    <CanceledIcon height='14px' width='14px' stroke={color} />
                </button>
                <Text
                    UNSAFE_style={{
                        fontSize: '12px',
                        marginBottom: '4px',
                        marginRight: '4px',
                        textAlign: 'center',
                        color,
                        textOverflow: 'ellipsis',
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                    }}
                >
                    {message}
                </Text>
            </Flex>
            <ProgressBar value={progress} aria-label={message} width='100px' showValueLabel={false} />
        </div>
    );
};

const useCurrentJob = () => {
    const { data: jobsData } = $api.useSuspenseQuery('get', '/api/jobs', undefined, {
        refetchInterval: 5000,
    });

    const { projectId } = useProjectIdentifier();
    const runningJob = jobsData.jobs.find(
        (job: Job) => job.project_id === projectId && (job.status === 'running' || job.status === 'pending')
    );

    return runningJob;
};

export const ProgressBarItem = () => {
    const trainingJob = useCurrentJob();

    if (trainingJob !== undefined) {
        return <TrainingStatusItem trainingJob={trainingJob} />;
    }

    return <IdleItem />;
};

export const Footer = () => {
    return (
        <View gridArea={'footer'} backgroundColor={'gray-100'} width={'100%'} height={'size-400'} overflow={'hidden'}>
            <Suspense>
                <ProgressBarItem />
            </Suspense>
        </View>
    );
};
