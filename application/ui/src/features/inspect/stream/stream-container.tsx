// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Button, Flex, Loading, toast, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';
import { useEnablePipeline, usePipeline } from 'src/hooks/use-pipeline.hook';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { Stream } from './stream';

import classes from '../inference.module.scss';

export const StreamContainer = () => {
    const { data: pipeline } = usePipeline();
    const { start, status } = useWebRTCConnection();
    const enablePipeline = useEnablePipeline({ onSuccess: start });
    const [size, setSize] = useState({ height: 608, width: 892 });

    const hasNoSink = isEmpty(pipeline?.sink);
    const hasNoSource = isEmpty(pipeline?.source);
    const isPipelineRunning = pipeline?.status === 'running';

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: 'Failed to connect to the stream' });
        }
    }, [status]);

    const handleStart = async () => {
        try {
            if (!isPipelineRunning) {
                await enablePipeline.mutate({ params: { path: { project_id: pipeline.project_id } } });
            }
            start();
        } catch (_error) {
            toast({ type: 'error', message: 'Failed to start the stream' });
        }
    };

    return (
        <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
            {status === 'idle' && (
                <div className={classes.canvasContainer}>
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
                </div>
            )}

            {(status === 'connecting' || enablePipeline.isPending) && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='90%' height='90%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Loading mode='inline' />
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connected' && (
                <div className={classes.canvasContainer}>
                    <Stream size={size} setSize={setSize} />
                </div>
            )}
        </View>
    );
};
