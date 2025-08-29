// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Button, Flex, Loading, toast, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { Stream } from './stream';

import classes from '../inference.module.scss';

export const StreamContainer = () => {
    const [size, setSize] = useState({ height: 608, width: 892 });
    const { start, status } = useWebRTCConnection();

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: 'Failed to connect to the stream' });
        }
    }, [status]);

    return (
        <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
            {status === 'idle' && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='90%' height='90%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Button onPress={start} UNSAFE_className={classes.playButton} aria-label={'Start stream'}>
                                <Play width='128px' height='128px' />
                            </Button>
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connecting' && (
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
