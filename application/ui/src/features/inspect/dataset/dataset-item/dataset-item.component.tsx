import { Image } from '@geti-inspect/icons';
import { Flex } from '@geti/ui';
import { clsx } from 'clsx';

import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
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
    mediaItem: MediaItem;
}

const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    const { selectedMediaItem, onSetSelectedMediaItem } = useSelectedMediaItem();

    const isSelected = selectedMediaItem?.id === mediaItem.id;

    const mediaUrl = `/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail`;

    const handleSelectMediaItem = () => {
        onSetSelectedMediaItem(mediaItem);
    };

    return (
        <div
            className={clsx(styles.datasetItem, {
                [styles.datasetItemSelected]: isSelected,
            })}
            onClick={handleSelectMediaItem}
        >
            <img src={mediaUrl} alt={mediaItem.filename} />
        </div>
    );
};

interface DatasetItemContainerProps {
    mediaItem: MediaItem | undefined;
}

export const DatasetItemContainer = ({ mediaItem }: DatasetItemContainerProps) => {
    if (mediaItem === undefined) {
        return <DatasetItemPlaceholder />;
    }

    return <DatasetItem mediaItem={mediaItem} />;
};
