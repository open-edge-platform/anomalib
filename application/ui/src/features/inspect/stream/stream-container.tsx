// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Flex, Loading, Text, View } from '@geti/ui';
import { Play, Refresh } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';
import { useActivatePipeline, usePipeline } from 'src/hooks/use-pipeline.hook';

import { useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { useAutoPlayStream } from './hook/use-auto-play-stream.hook';
import { Stream } from './stream';

import classes from './stream-container.module.scss';

const RECONNECT_CLEANUP_DELAY_MS = 300; // Delay to allow stream connection cleanup to complete before reconnecting

export const StreamContainer = () => {
    const { projectId } = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const { start, stop, status } = useStreamConnection();
    const activePipeline = useActivatePipeline({ onSuccess: start });

    useAutoPlayStream();

    const hasSource = !isEmpty(pipeline?.source);

    const handleStart = () => {
        activePipeline.mutate({ params: { path: { project_id: projectId } } });
    };

    const handleReconnect = async () => {
        try {
            // Stop the old connection first to clean it up
            await stop();
            // Wait for cleanup to complete and status to update to 'idle'
            await new Promise((resolve) => setTimeout(resolve, RECONNECT_CLEANUP_DELAY_MS));

            // If pipeline is already running, just start the stream directly
            // Otherwise, activate the pipeline which will start streaming via onSuccess callback
            if (pipeline?.status === 'running') {
                await start();
            } else {
                activePipeline.mutate({ params: { path: { project_id: projectId } } });
            }
        } catch (error) {
            console.error('Failed to reconnect stream:', error);
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
                            isDisabled={!hasSource || activePipeline.isPending}
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

            {(status === 'disconnected' || status === 'failed') && (
                <View backgroundColor={'gray-200'} width='90%' height='90%'>
                    <Flex
                        alignItems={'center'}
                        justifyContent={'center'}
                        height='100%'
                        direction='column'
                        gap='size-200'
                    >
                        <Text>Stream disconnected</Text>
                        <Button
                            onPress={handleReconnect}
                            aria-label={'Reconnect stream'}
                            isDisabled={activePipeline.isPending}
                            variant='primary'
                        >
                            <Refresh />
                            Reconnect
                        </Button>
                    </Flex>
                </View>
            )}

            {(status === 'connecting' || status === 'connected') && <Stream />}
        </Flex>
    );
};
