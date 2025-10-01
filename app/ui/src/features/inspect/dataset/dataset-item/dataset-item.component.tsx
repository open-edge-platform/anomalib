import { Image } from '@geti-inspect/icons';
import { Flex, View } from '@geti/ui';
import { clsx } from 'clsx';

import { type MediaItem } from '../types';

import styles from './dataset-item.module.scss';

const DatasetItemPlaceholder = () => {
    return (
        <Flex
            justifyContent={'center'}
            alignItems={'center'}
            UNSAFE_className={clsx(styles.datasetItemPlaceholder, styles.datasetItem)}
        >
            <Flex>
                <Image />
            </Flex>
        </Flex>
    );
};

interface DatasetItemProps {
    mediaItem: MediaItem | undefined;
}

export const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    if (mediaItem === undefined) {
        return <DatasetItemPlaceholder />;
    }

    // TODO: Replace with thumbnail once supported by the backend
    const mediaUrl = `/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/full`;

    return (
        <View UNSAFE_className={styles.datasetItem}>
            <img src={mediaUrl} alt={mediaItem.filename} />
        </View>
    );
};
