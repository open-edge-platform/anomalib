import { Heading, View } from '@geti/ui';

import { useInference } from './inference-provider.component';
import { useSelectedMediaItem } from './selected-media-item-provider.component';

import styles from './inference.module.scss';

export const ImageInference = () => {
    const { selectedMediaItem } = useSelectedMediaItem();
    const { isPending, inferenceResult } = useInference();

    if (selectedMediaItem === undefined) {
        return (
            <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer}>
                <Heading>Select an image to start inference</Heading>
            </View>
        );
    }

    if (!isPending || inferenceResult === undefined) {
        const mediaUrl = `/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`;

        return (
            <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer}>
                <img src={mediaUrl} alt={selectedMediaItem.filename} className={styles.inferencedImage} />
            </View>
        );
    }

    return null;

    /*return (
        <View gridArea={'canvas'} UNSAFE_className={styles.canvasContainer}>
            <img src={mediaUrl} alt={selectedMediaItem.filename} />
        </View>
    );*/
};
