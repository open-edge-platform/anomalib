// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { $api } from '@geti-inspect/api';
import { SchemaJob as Job } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';

import { useStatusBar } from '../status-bar-context';

export const TrainingStatusAdapter = () => {
    const { setStatus, removeStatus } = useStatusBar();
    const { projectId } = useProjectIdentifier();

    const { data: jobsData } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: 5000,
    });

    const { mutate: cancelJob } = $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });

    const trainingJob = jobsData?.jobs.find(
        (job: Job) =>
            job.project_id === projectId &&
            job.type === 'training' &&
            (job.status === 'running' || job.status === 'pending')
    );

    useEffect(() => {
        if (!trainingJob) {
            removeStatus('training');
            return;
        }

        const jobId = trainingJob.id;
        if (!jobId) {
            return;
        }

        setStatus({
            id: 'training',
            type: 'training',
            message: `Training ${trainingJob.payload.model_name}...`,
            detail: trainingJob.message,
            progress: trainingJob.progress,
            variant: 'info',
            isCancellable: true,
            onCancel: () => {
                if (trainingJob.id) {
                    cancelJob({ params: { path: { job_id: trainingJob.id } } });
                }
            },
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        trainingJob?.id,
        trainingJob?.payload.model_name,
        trainingJob?.message,
        trainingJob?.progress,
        trainingJob?.status,
        setStatus,
        removeStatus,
        cancelJob,
    ]);

    return null;
};
