import { Grid, Heading, minmax, repeat } from '@geti/ui';

import { DatasetItem } from './dataset-item/dataset-item.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface DatasetItemProps {
    mediaItems: MediaItem[];
}

export const DatasetList = ({ mediaItems }: DatasetItemProps) => {
    const mediaItemsToRender = [
        ...mediaItems,
        ...Array.from({
            length: Math.max(0, REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItems.length),
        }).map(() => undefined),
    ];

    return (
        <Grid
            flex={1}
            columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
            rows={['max-content', '1fr']}
            gap={'size-100'}
            alignContent={'start'}
        >
            <Heading gridColumn={'1/-1'}>Normal images</Heading>
            {mediaItemsToRender.map((mediaItem, index) => (
                <DatasetItem key={mediaItem?.id ?? index} mediaItem={mediaItem} />
            ))}
        </Grid>
    );
};
