import { useState } from 'react';

import { Skeleton } from '@geti/ui';
import { clsx } from 'clsx';

import { DeleteMediaItem } from '../delete-dataset-item/delete-dataset-item.component';
import { type MediaItem } from '../types';

import styles from './dataset-item.module.scss';

interface DatasetItemProps {
    isSelected: boolean;
    mediaItem: MediaItem;
    onClick: () => void;
    onDeleted: () => void;
}

const RETRY_LIMIT = 3;

export const DatasetItem = ({ isSelected, mediaItem, onClick, onDeleted }: DatasetItemProps) => {
    const [retry, setRetry] = useState(0);
    const [isLoading, setIsLoading] = useState(true);

    const mediaUrl = `/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail?retry=${retry}`;

    const handleError = () => {
        if (retry < RETRY_LIMIT) {
            setRetry((current) => current + 1);
            setIsLoading(true);
        } else {
            setIsLoading(false);
        }
    };

    const handleLoad = () => {
        setIsLoading(false);
    };

    return (
        <div className={clsx(styles.datasetItem, { [styles.datasetItemSelected]: isSelected })} onClick={onClick}>
            {isLoading && <Skeleton width={'100%'} height={'100%'} UNSAFE_className={styles.loader} />}

            <img src={mediaUrl} alt={mediaItem.filename} onError={handleError} onLoad={handleLoad} />
            <div className={clsx(styles.floatingContainer, styles.rightTopElement)}>
                <DeleteMediaItem itemsIds={[String(mediaItem.id)]} onDeleted={onDeleted} />
            </div>
        </div>
    );
};
