import { useEffect } from 'react';

import { Skeleton } from '@geti/ui';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';

import { useInference } from '../../inference-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { DeleteMediaItem } from '../delete-dataset-item/delete-dataset-item.component';
import { type MediaItem } from '../types';

import styles from './dataset-item.module.scss';

interface DatasetItemProps {
    mediaItem: MediaItem;
}

export const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    const { selectedMediaItem, onSetSelectedMediaItem } = useSelectedMediaItem();
    const { onInference, selectedModelId } = useInference();

    const isSelected = selectedMediaItem?.id === mediaItem.id;

    const { data: thumbnailBlob, isLoading } = useQuery({
        queryKey: ['media', mediaItem.id],
        queryFn: async () => {
            const response = await fetch(`/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail`);

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return URL.createObjectURL(await response.blob());
        },
        retry: 3,
        retryDelay: (attemptIndex: number) => Math.min(1000 * 2 ** attemptIndex, 10000),
    });

    useEffect(() => {
        return () => {
            thumbnailBlob && URL.revokeObjectURL(thumbnailBlob);
        };
    }, [thumbnailBlob]);

    const handleClick = async () => {
        const selection = mediaItem.id === selectedMediaItem?.id ? undefined : mediaItem;

        onSetSelectedMediaItem(selection);
        selectedModelId !== undefined && (await onInference(mediaItem, selectedModelId));
    };

    return (
        <div className={clsx(styles.datasetItem, { [styles.datasetItemSelected]: isSelected })} onClick={handleClick}>
            {isLoading || !thumbnailBlob ? (
                <Skeleton width={'100%'} height={'100%'} />
            ) : (
                <>
                    <img src={thumbnailBlob} alt={mediaItem.filename} />
                    <div className={clsx(styles.floatingContainer, styles.rightTopElement)}>
                        <DeleteMediaItem
                            itemsIds={[String(mediaItem.id)]}
                            onDeleted={() => {
                                selectedMediaItem?.id === mediaItem.id && onSetSelectedMediaItem(undefined);
                            }}
                        />
                    </div>
                </>
            )}
        </div>
    );
};
