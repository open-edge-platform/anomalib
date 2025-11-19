import { Grid, Heading } from '@geti/ui';

import styles from './source-sink-message.module.scss';

export const SOURCE_MESSAGE = 'No source configured. Please set it before starting the stream.';

export const SourceSinkMessage = () => {
    return (
        <Grid
            gridArea={'canvas'}
            UNSAFE_className={styles.canvasContainer}
            justifyContent={'center'}
            alignContent={'center'}
        >
            <Heading>{SOURCE_MESSAGE}</Heading>
        </Grid>
    );
};
