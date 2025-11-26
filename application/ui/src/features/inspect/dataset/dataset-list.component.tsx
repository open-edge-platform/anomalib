import { DialogContainer, Flex, Grid, Heading, minmax, repeat } from '@geti/ui';
import isEmpty from 'lodash-es/isEmpty';

import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { DatasetItemPlaceholder } from './dataset-item/dataset-item-placeholder.component';
import { DatasetItem } from './dataset-item/dataset-item.component';
import { MediaPreview } from './media-preview/media-preview.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface DatasetItemProps {
    mediaItems: MediaItem[];
}

export const DatasetList = ({ mediaItems }: DatasetItemProps) => {
    const { selectedMediaItem, onSetSelectedMediaItem } = useSelectedMediaItem();

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
                            onClick={() => onSetSelectedMediaItem(mediaItem)}
                            onDeleted={() =>
                                selectedMediaItem?.id === mediaItem.id && onSetSelectedMediaItem(undefined)
                            }
                        />
                    )
                )}
            </Grid>

            <DialogContainer onDismiss={() => onSetSelectedMediaItem(undefined)}>
                {!isEmpty(selectedMediaItem) && (
                    <MediaPreview
                        mediaItems={mediaItems}
                        selectedMediaItem={selectedMediaItem}
                        onSetSelectedMediaItem={onSetSelectedMediaItem}
                        onClose={() => onSetSelectedMediaItem(undefined)}
                    />
                )}
            </DialogContainer>
        </Flex>
    );
};
