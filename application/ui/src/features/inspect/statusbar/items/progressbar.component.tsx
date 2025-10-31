import React, { useEffect, useState } from 'react';

import { $api, API_BASE_URL } from '@geti-inspect/api';
import { SchemaJob as Job } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, ProgressBar, Text } from '@geti/ui';
import { CanceledIcon, WaitingIcon } from '@geti/ui/icons';

function IdleItem(): React.ReactNode {
    return (
        <Flex
            direction='row'
            alignItems='center'
            width='100px'
            justifyContent='space-between'
            UNSAFE_style={{ padding: '5px' }}
        >
            <WaitingIcon height='14px' width='14px' stroke='var(--spectrum-global-color-gray-600)' />
            <Text marginStart={'5px'} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                Idle
            </Text>
        </Flex>
    );
}

function TrainingStatusItem(progress: number, stage: string, onCancel?: () => void): React.ReactNode {
    // Determine color based on stage
    let bgcolor = 'var(--spectrum-global-color-blue-600)';
    let fgcolor = '#fff';
    if (stage.toLowerCase().includes('valid')) {
        bgcolor = 'var(--spectrum-global-color-yellow-600)';
        fgcolor = '#000';
    } else if (stage.toLowerCase().includes('test')) {
        bgcolor = 'var(--spectrum-global-color-green-600)';
        fgcolor = '#fff';
    } else if (stage.toLowerCase().includes('train') || stage.toLowerCase().includes('fit')) {
        bgcolor = 'var(--spectrum-global-color-blue-600)';
        fgcolor = '#fff';
    }

    return (
        <div
            style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: bgcolor,
            }}
        >
            <Flex direction='row' alignItems='center' width='100px' justifyContent='space-between'>
                <button
                    onClick={() => {
                        console.info('Cancel training');
                        if (onCancel) {
                            onCancel();
                        }
                    }}
                    style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                    }}
                >
                    <CanceledIcon height='14px' width='14px' stroke={fgcolor} />
                </button>
                <Text
                    UNSAFE_style={{
                        fontSize: '12px',
                        marginBottom: '4px',
                        marginRight: '4px',
                        textAlign: 'center',
                        color: fgcolor,
                    }}
                >
                    {stage}
                </Text>
            </Flex>
            <ProgressBar value={progress} aria-label={stage} width='100px' showValueLabel={false} />
        </div>
    );
}

function getElement(status: string, stage: string, progress: number, onCancel?: () => void): React.ReactNode {
    if (status === 'running') {
        return TrainingStatusItem(progress, stage, onCancel);
    }
    return IdleItem();
}

export const ProgressBarItem = () => {
    const { projectId } = useProjectIdentifier();
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [jobStatus, setJobStatus] = useState<string>('idle');
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [sseConnected, setSSEConnected] = useState(false);

    // Fetch the current running job for the project
    const { data: jobsData } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: 5000, // Refetch every 5 seconds to check for new jobs
    });

    const cancelJobMutation = $api.useMutation('post', '/api/jobs/{job_id}:cancel');

    // Find the running or pending job for this project
    useEffect(() => {
        if (!jobsData?.jobs) {
            return;
        }

        const runningJob = jobsData.jobs.find(
            (job: Job) => job.project_id === projectId && (job.status === 'running' || job.status === 'pending')
        );

        if (runningJob) {
            setCurrentJobId(runningJob.id ?? null);
            setJobStatus(runningJob.status);

            // Only use polling data if SSE is not connected (fallback)
            if (!sseConnected) {
                setProgress(runningJob.progress);
                setStage(runningJob.stage || runningJob.status);
            }
        } else {
            // No running job found - reset everything
            setCurrentJobId(null);
            setJobStatus('idle');
            setProgress(0);
            setStage('');
            setSSEConnected(false);
        }
    }, [jobsData, projectId, sseConnected]);

    // Connect to SSE for progress updates when there's a running job
    useEffect(() => {
        if (!currentJobId || jobStatus !== 'running') {
            setSSEConnected(false);
            return;
        }

        const eventSource = new EventSource(`${API_BASE_URL}/api/jobs/${currentJobId}/progress`);

        eventSource.onopen = () => {
            setSSEConnected(true);
            console.debug('SSE connected');
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.debug('SSE data:', data);
                if (data.progress !== undefined) {
                    setProgress(data.progress);
                }
                if (data.stage !== undefined) {
                    setStage(data.stage);
                }
            } catch (error) {
                console.error('Failed to parse progress data:', error);
            }
        };

        eventSource.onerror = (error) => {
            setSSEConnected(false);
            eventSource.close();
        };

        return () => {
            setSSEConnected(false);
            eventSource.close();
        };
    }, [currentJobId, jobStatus]);

    const handleCancel = async () => {
        if (!currentJobId) {
            return;
        }

        try {
            await cancelJobMutation.mutateAsync({
                params: {
                    path: {
                        job_id: currentJobId,
                    },
                },
            });
            console.info('Job cancelled successfully');
        } catch (error) {
            console.error('Failed to cancel job:', error);
        }
    };

    return getElement(jobStatus, stage, progress, handleCancel);
};
