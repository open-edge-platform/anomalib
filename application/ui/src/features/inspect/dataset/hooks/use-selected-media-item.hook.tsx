import { useEffect } from 'react';

import { isNil } from 'lodash-es';
import { useQueryState } from 'nuqs';

import { useGetMediaItems } from './use-get-media-items.hook';

export const useSelectedMediaItem = () => {
    const { mediaItems, isFetchingNextPage, hasNextPage, fetchNextPage } = useGetMediaItems();
    const [selectedMediaItemId, setSelectedMediaItemId] = useQueryState('selectedMediaItemId');

    const selectedMediaItem = mediaItems.find((item) => item.id === selectedMediaItemId);

    useEffect(() => {
        if (isNil(selectedMediaItemId) || !isNil(selectedMediaItem) || isFetchingNextPage || !hasNextPage) {
            return;
        }

        fetchNextPage();
    }, [fetchNextPage, hasNextPage, isFetchingNextPage, selectedMediaItem, selectedMediaItemId]);

    return [selectedMediaItem, setSelectedMediaItemId] as const;
};
