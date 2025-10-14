import { Heading, Loading, View } from '@geti/ui';
import { clsx } from 'clsx';

import { useInference } from './inference-provider.component';
import { useSelectedMediaItem } from './selected-media-item-provider.component';

import styles from './inference.module.scss';

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
                    className={clsx(styles.inferencedImage, { [styles.notReadyInference]: isPending })}
                />
                <Loading mode={'overlay'} />
            </View>
        );
    }

    const src = `data:image/png;base64,${inferenceResult.anomaly_map}`;

    return (
        <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer}>
            <img src={src} alt={selectedMediaItem.filename} className={styles.inferencedImage} />
        </View>
    );
};
