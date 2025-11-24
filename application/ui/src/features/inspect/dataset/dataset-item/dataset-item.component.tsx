import { useState } from 'react';

import { Skeleton } from '@geti/ui';
import { clsx } from 'clsx';

import { useInference } from '../../inference-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { DeleteMediaItem } from '../delete-dataset-item/delete-dataset-item.component';
import { type MediaItem } from '../types';

import styles from './dataset-item.module.scss';

interface DatasetItemProps {
    mediaItem: MediaItem;
}

const RETRY_LIMIT = 3;

export const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    const [retry, setRetry] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const { onInference, selectedModelId } = useInference();
    const { selectedMediaItem, onSetSelectedMediaItem } = useSelectedMediaItem();

    const isSelected = selectedMediaItem?.id === mediaItem.id;

    const mediaUrl = `/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail?retry=${retry}`;

    const handleError = () => {
        if (retry < RETRY_LIMIT) {
            setRetry((current) => current + 1);
        }
    };

    const handleLoad = () => {
        setIsLoading(false);
    };

    const handleClick = async () => {
        const selection = mediaItem.id === selectedMediaItem?.id ? undefined : mediaItem;

        onSetSelectedMediaItem(selection);
        selectedModelId !== undefined && (await onInference(mediaItem, selectedModelId));
    };

    return (
        <div className={clsx(styles.datasetItem, { [styles.datasetItemSelected]: isSelected })} onClick={handleClick}>
            {isLoading && <Skeleton width={'100%'} height={'100%'} UNSAFE_className={styles.loader} />}

            <img src={mediaUrl} alt={mediaItem.filename} onError={handleError} onLoad={handleLoad} />
            <div className={clsx(styles.floatingContainer, styles.rightTopElement)}>
                <DeleteMediaItem
                    itemsIds={[String(mediaItem.id)]}
                    onDeleted={() => {
                        selectedMediaItem?.id === mediaItem.id && onSetSelectedMediaItem(undefined);
                    }}
                />
            </div>
        </div>
    );
};
