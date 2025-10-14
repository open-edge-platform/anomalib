import { Flex, Text } from '@adobe/react-spectrum';
import { Grid, Heading, Loading, View } from '@geti/ui';
import { clsx } from 'clsx';

import { useInference } from './inference-provider.component';
import { useSelectedMediaItem } from './selected-media-item-provider.component';

import styles from './inference.module.scss';

interface LabelProps {
    label: string;
    score: number;
}

const LabelScore = ({ label, score }: LabelProps) => {
    const formatter = new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 0,
        style: 'percent',
    });

    return (
        <Flex
            gridArea={'label'}
            UNSAFE_className={clsx(styles.label, {
                [styles.labelNormal]: label.toLowerCase() === 'normal',
                [styles.labelAnomalous]: label.toLowerCase() === 'anomalous',
            })}
            gap={'size-50'}
        >
            <Text>{label}</Text>
            <Text>{formatter.format(score)}</Text>
        </Flex>
    );
};

export const InferenceResult = () => {
    const { selectedMediaItem } = useSelectedMediaItem();
    const { isPending, inferenceResult } = useInference();

    if (selectedMediaItem === undefined) {
        return (
            <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer}>
                <Heading>Select an image to start inference</Heading>
            </View>
        );
    }

    if (isPending || inferenceResult === undefined) {
        const mediaUrl = `/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`;

        return (
            <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer} position={'relative'}>
                <img
                    src={mediaUrl}
                    alt={selectedMediaItem.filename}
                    className={clsx(styles.img, { [styles.notReadyInference]: isPending })}
                />
                <Loading mode={'overlay'} />
            </View>
        );
    }

    const src = `data:image/png;base64,${inferenceResult.anomaly_map}`;

    return (
        <Grid gridArea={'canvas'} UNSAFE_className={clsx(styles.canvasContainer, styles.inferenceResultContainer)}>
            <LabelScore label={inferenceResult.label} score={inferenceResult.score} />
            <img src={src} alt={selectedMediaItem.filename} className={clsx(styles.img, styles.inferencedImage)} />
        </Grid>
    );
};
