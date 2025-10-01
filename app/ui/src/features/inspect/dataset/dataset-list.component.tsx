import { Grid, minmax, repeat } from '@geti/ui';

import { DatasetItem } from './dataset-item/dataset-item.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface DatasetItemProps {
    mediaItems: MediaItem[];
}

export const DatasetList = ({ mediaItems }: DatasetItemProps) => {
    const mediaItemsToRender =
        mediaItems.length >= REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING
            ? mediaItems
            : Array.from({ length: REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING }).map((_, index) =>
                  index <= mediaItems.length - 1 ? mediaItems[index] : undefined
              );

    return (
        <Grid
            flex={1}
            columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
            gap={'size-100'}
            alignContent={'start'}
        >
            {mediaItemsToRender.map((mediaItem, index) => (
                <DatasetItem key={mediaItem?.id ?? index} mediaItem={mediaItem} />
            ))}
        </Grid>
    );
};
