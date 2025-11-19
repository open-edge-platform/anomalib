// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Flex, Loading, toast, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';
import { useActivatePipeline, usePipeline } from 'src/hooks/use-pipeline.hook';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { isStatusActive } from '../utils';
import { Stream } from './stream';

import classes from './stream-container.module.scss';

export const StreamContainer = () => {
    const { projectId } = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const { start, status } = useWebRTCConnection();
    const activePipeline = useActivatePipeline({ onSuccess: start });
    const [size, setSize] = useState({ height: 608, width: 892 });

    const hasNoSink = isEmpty(pipeline?.sink);
    const hasNoSource = isEmpty(pipeline?.source);
    const isPipelineActive = isStatusActive(pipeline.status);

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: 'Failed to connect to the stream' });
        }

        if (isPipelineActive && status === 'idle') {
            start();
        }
    }, [isPipelineActive, status, start]);

    const handleStart = async () => {
        if (isPipelineActive) {
            start();
        } else {
            activePipeline.mutate({ params: { path: { project_id: projectId } } });
        }
    };

    return (
        <Flex
            gridArea={'canvas'}
            maxHeight={'100%'}
            UNSAFE_className={classes.canvasContainer}
            alignItems={'center'}
            justifyContent={'center'}
        >
            {status === 'idle' && (
                <View backgroundColor={'gray-200'} width='90%' height='90%'>
                    <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                        <Button
                            onPress={handleStart}
                            aria-label={'Start stream'}
                            isDisabled={hasNoSink || hasNoSource}
                            UNSAFE_className={classes.playButton}
                        >
                            <Play width='128px' height='128px' />
                        </Button>
                    </Flex>
                </View>
            )}

            {(status === 'connecting' || activePipeline.isPending) && (
                <View backgroundColor={'gray-200'} width='90%' height='90%'>
                    <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                        <Loading mode='inline' />
                    </Flex>
                </View>
            )}

            {status === 'connected' && <Stream size={size} setSize={setSize} />}
        </Flex>
    );
};
