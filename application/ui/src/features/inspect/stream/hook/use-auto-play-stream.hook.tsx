// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { toast } from '@geti/ui';
import { usePipeline, useRunPipeline } from 'src/hooks/use-pipeline.hook';

import { useWebRTCConnection } from '../../../../components/stream/web-rtc-connection-provider';
import { isNonEmptyString } from '../../utils';

export const STREAM_ERROR_MESSAGE = 'Failed to connect to the stream';

export const useAutoPlayStream = () => {
    const runPipeline = useRunPipeline({});
    const { data: pipeline } = usePipeline();
    const { start, status } = useWebRTCConnection();
    const { projectId } = useProjectIdentifier();

    const hasSink = isNonEmptyString(pipeline?.sink?.id);
    const hasModel = isNonEmptyString(pipeline?.model?.id);
    const hasSource = isNonEmptyString(pipeline?.source?.id);
    const isRunning = pipeline?.status === 'running';
    const hasInferenceConfig = hasModel && hasSource && hasSink;

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: STREAM_ERROR_MESSAGE });
        }

        if (hasSource && status === 'idle') {
            start();
        }

        if (hasInferenceConfig && !isRunning && !runPipeline.isPending) {
            runPipeline.mutate({ params: { path: { project_id: projectId } } });
        }
    }, [hasSource, status, start, hasInferenceConfig, isRunning, runPipeline, projectId]);
};
