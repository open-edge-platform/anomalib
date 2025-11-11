import { Flex, StatusLight } from '@geti/ui';

import { useWebRTCConnection } from './web-rtc-connection-provider';

export const WebRTCConnectionStatus = () => {
    const { status } = useWebRTCConnection();

    switch (status) {
        case 'idle':
            return (
                <Flex
                    gap='size-100'
                    alignItems={'center'}
                    UNSAFE_style={{
                        '--spectrum-gray-visual-color': 'var(--spectrum-global-color-gray-500)',
                    }}
                >
                    <StatusLight role={'status'} aria-label='Idle' variant='neutral'>
                        Idle
                    </StatusLight>
                </Flex>
            );
        case 'connecting':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connecting' variant='info'>
                        Connecting
                    </StatusLight>
                </Flex>
            );
        case 'disconnected':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Disconnected' variant='negative'>
                        Disconnected
                    </StatusLight>
                </Flex>
            );
        case 'failed':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Failed' variant='negative'>
                        Failed
                    </StatusLight>
                </Flex>
            );
        case 'connected':
            return (
                <Flex gap='size-200' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connected' variant='positive'>
                        Connected
                    </StatusLight>
                </Flex>
            );
    }
};
