import { useEffect } from 'react';

import { LinkExpired } from '@geti-inspect/icons';
import { Button, Flex, Text } from '@geti/ui';

import { useWebRTCConnection } from '../../../../components/stream/web-rtc-connection-provider';

import classes from './enable-project.module.scss';

export const EnableProject = () => {
    const { stop } = useWebRTCConnection();

    useEffect(() => {
        stop();
    }, [stop]);

    return (
        <Flex UNSAFE_className={classes.container} alignItems={'center'} justifyContent={'center'}>
            <Flex direction='column' width={'90%'} maxWidth={'32rem'} gap={'size-200'} alignItems={'center'}>
                <LinkExpired />

                <Text UNSAFE_className={classes.description}>
                    This project is set as inactive, therefore the pipeline configuration is disabled for this project.
                    You can still explore the dataset and models within this inactive project.
                </Text>

                <Text UNSAFE_className={classes.title}>Would you like to activate this project?</Text>

                <Button onPress={() => console.log('Activate project')}>Activate project</Button>
            </Flex>
        </Flex>
    );
};
