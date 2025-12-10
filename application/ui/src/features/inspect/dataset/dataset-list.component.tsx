import { DialogContainer, Flex, Heading, Selection, Size, View } from '@geti/ui';
import { isNil } from 'lodash-es';
import isEmpty from 'lodash-es/isEmpty';
import { useQueryState } from 'nuqs';
import { MediaThumbnail } from 'src/components/media-thumbnail/media-thumbnail.component';
import { GridMediaItem } from 'src/components/virtualizer-grid-layout/grid-media-item/grid-media-item.component';
import { VirtualizerGridLayout } from 'src/components/virtualizer-grid-layout/virtualizer-grid-layout.component';

import { getThumbnailUrl } from '../utils';
import { DatasetItemPlaceholder } from './dataset-item-placeholder/dataset-item-placeholder.component';
import { getPlaceholderItem, isPlaceholderItem } from './dataset-item-placeholder/util';
import { DeleteMediaItem } from './delete-dataset-item/delete-dataset-item.component';
import { DownloadDatasetItem } from './download-dataset-item/download-dataset-item.component';
import { useGetMediaItems } from './hooks/use-get-media-items.hook';
import { MediaPreview } from './media-preview/media-preview.component';
import { InferenceOpacityProvider } from './media-preview/providers/inference-opacity-provider.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

const layoutOptions = {
    maxColumns: 4,
    minSpace: new Size(8, 8),
    minItemSize: new Size(120, 120),
    preserveAspectRatio: true,
};

export const DatasetList = () => {
    const { mediaItems, isFetchingNextPage, fetchNextPage } = useGetMediaItems();
    const [selectedMediaItemId, setSelectedMediaItem] = useQueryState('selectedMediaItem');
    //TODO: revisit implementation when dataset loading is paginated
    const selectedMediaItem = mediaItems.find((item) => item.id === selectedMediaItemId);

    const mediaItemsToRender = [
        ...mediaItems,
        ...Array.from({
            length: Math.max(0, REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItems.length),
        }).map((_, index): MediaItem => getPlaceholderItem(index)),
    ];

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;
        const itemId = String(firstKey);

        if (!isNil(firstKey) && !isPlaceholderItem(itemId)) {
            setSelectedMediaItem(itemId);
        }
    };

    return (
        <Flex gap='size-200' direction={'column'} height={'100%'}>
            <Heading>Normal images</Heading>

            <View width={'100%'} height={'100%'}>
                <VirtualizerGridLayout
                    items={mediaItemsToRender}
                    ariaLabel='sidebar-items'
                    selectionMode='single'
                    layoutOptions={layoutOptions}
                    isLoadingMore={isFetchingNextPage}
                    onLoadMore={fetchNextPage}
                    onSelectionChange={handleSelectionChange}
                    contentItem={(mediaItem) =>
                        mediaItem.filename === 'placeholder' ? (
                            <DatasetItemPlaceholder />
                        ) : (
                            <GridMediaItem
                                contentElement={() => (
                                    <MediaThumbnail
                                        alt={mediaItem.filename}
                                        url={getThumbnailUrl(mediaItem)}
                                        onClick={() => setSelectedMediaItem(mediaItem.id ?? null)}
                                    />
                                )}
                                topRightElement={() => (
                                    <DeleteMediaItem
                                        itemsIds={[String(mediaItem.id)]}
                                        onDeleted={() => setSelectedMediaItem(null)}
                                    />
                                )}
                                bottomLeftElement={() => <DownloadDatasetItem mediaItem={mediaItem} />}
                            />
                        )
                    }
                />
            </View>

            <DialogContainer onDismiss={() => setSelectedMediaItem(null)}>
                {!isEmpty(selectedMediaItem) && (
                    <InferenceOpacityProvider>
                        <MediaPreview
                            mediaItems={mediaItems}
                            selectedMediaItem={selectedMediaItem}
                            onClose={() => setSelectedMediaItem(null)}
                            onSelectedMediaItem={setSelectedMediaItem}
                        />
                    </InferenceOpacityProvider>
                )}
            </DialogContainer>
        </Flex>
    );
};
