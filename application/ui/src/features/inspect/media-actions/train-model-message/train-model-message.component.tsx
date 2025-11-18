import { Grid, Heading } from '@geti/ui';

import styles from './train-model-message.module.scss';

export const TrainModelMessage = () => {
    return (
        <Grid
            gridArea={'canvas'}
            UNSAFE_className={styles.canvasContainer}
            justifyContent={'center'}
            alignContent={'center'}
        >
            <Heading>No trained models available. Please train a model to start inference.</Heading>
        </Grid>
    );
};
