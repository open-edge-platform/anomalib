import { Grid, Heading } from '@geti/ui';

import styles from './source-sink-message.module.scss';

export const SourceSinkMessage = () => {
    return (
        <Grid
            gridArea={'canvas'}
            UNSAFE_className={styles.canvasContainer}
            justifyContent={'center'}
            alignContent={'center'}
        >
            <Heading>No source or sink is configured. Please set both before starting the stream.</Heading>
        </Grid>
    );
};
