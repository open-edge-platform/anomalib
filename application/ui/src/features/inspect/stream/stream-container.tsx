// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Flex, Loading, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';
import { useActivatePipeline, usePipeline } from 'src/hooks/use-pipeline.hook';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { useAutoPlayStream } from './hook/use-auto-play-stream.hook';
import { Stream } from './stream';

import classes from './stream-container.module.scss';

export const StreamContainer = () => {
    const { projectId } = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const { start, status } = useWebRTCConnection();
    const activePipeline = useActivatePipeline({ onSuccess: start });

    useAutoPlayStream();

    const hasSource = !isEmpty(pipeline?.source);

    const handleStart = () => {
        activePipeline.mutate({ params: { path: { project_id: projectId } } });
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

            {status === 'connected' && <Stream />}
        </Flex>
    );
};
