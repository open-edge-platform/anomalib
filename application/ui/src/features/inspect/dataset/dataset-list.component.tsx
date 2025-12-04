import { DialogContainer, Flex, Grid, Heading, minmax, repeat } from '@geti/ui';
import isEmpty from 'lodash-es/isEmpty';
import { useQueryState } from 'nuqs';

import { DatasetItemPlaceholder } from './dataset-item/dataset-item-placeholder.component';
import { DatasetItem } from './dataset-item/dataset-item.component';
import { MediaPreview } from './media-preview/media-preview.component';
import { InferenceOpacityProvider } from './media-preview/providers/inference-opacity-provider.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface DatasetItemProps {
    mediaItems: MediaItem[];
}

export const DatasetList = ({ mediaItems }: DatasetItemProps) => {
    const [selectedMediaItemId, setSelectedMediaItem] = useQueryState('selectedMediaItem');
    //TODO: revisit implementation when dataset loading is paginated
    const selectedMediaItem = mediaItems.find((item) => item.id === selectedMediaItemId);

    const mediaItemsToRender = [
        ...mediaItems,
        ...Array.from({
            length: Math.max(0, REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItems.length),
        }).map(() => undefined),
    ];

    return (
        <Flex gap='size-200' direction={'column'} height={'100%'}>
            <Heading>Normal images</Heading>

            <Grid
                flex={1}
                gap={'size-100'}
                rows={['max-content', '1fr']}
                alignContent={'start'}
                columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
            >
                {mediaItemsToRender.map((mediaItem, index) =>
                    isEmpty(mediaItem) ? (
                        <DatasetItemPlaceholder key={index} />
                    ) : (
                        <DatasetItem
                            key={mediaItem.id}
                            mediaItem={mediaItem}
                            isSelected={selectedMediaItem?.id === mediaItem.id}
                            onClick={() => setSelectedMediaItem(mediaItem?.id ?? null)}
                            onDeleted={() => selectedMediaItem?.id === mediaItem.id && setSelectedMediaItem(null)}
                        />
                    )
                )}
            </Grid>

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
