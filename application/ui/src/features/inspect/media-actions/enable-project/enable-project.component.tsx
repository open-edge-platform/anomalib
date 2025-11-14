import { Suspense, useEffect } from 'react';

import { LinkExpired } from '@geti-inspect/icons';
import { Button, DialogTrigger, Flex, Loading, Text } from '@geti/ui';

import { useWebRTCConnection } from '../../../../components/stream/web-rtc-connection-provider';
import { ConfirmationDialog } from './confirmation-dialog.component';

import classes from './enable-project.module.scss';

interface EnableProjectProps {
    activeProjectId: string;
    currentProjectId: string;
}

const useStopCurrentWebRtcConnection = () => {
    const { stop } = useWebRTCConnection();

    useEffect(() => {
        stop();
    }, [stop]);
};

export const EnableProject = ({ activeProjectId, currentProjectId }: EnableProjectProps) => {
    useStopCurrentWebRtcConnection();

    return (
        <Flex UNSAFE_className={classes.container} alignItems={'center'} justifyContent={'center'}>
            <Flex direction='column' width={'90%'} maxWidth={'32rem'} gap={'size-200'} alignItems={'center'}>
                <LinkExpired />

                <Text UNSAFE_className={classes.description}>
                    This project is set as inactive, therefore the pipeline configuration is disabled for this project.
                    You can still explore the dataset and models within this inactive project.
                </Text>

                <Text UNSAFE_className={classes.title}>Would you like to activate this project?</Text>

                <DialogTrigger>
                    <Button>Activate project</Button>
                    <Suspense fallback={<Loading mode={'inline'} />}>
                        <ConfirmationDialog activeProjectId={activeProjectId} currentProjectId={currentProjectId} />
                    </Suspense>
                </DialogTrigger>
            </Flex>
        </Flex>
    );
};
