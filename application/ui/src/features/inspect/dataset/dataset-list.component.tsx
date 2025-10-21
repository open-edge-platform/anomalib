import { Flex, Grid, Heading, minmax, repeat } from '@geti/ui';

import { DatasetItemContainer } from './dataset-item/dataset-item.component';
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
        <Flex gap='size-200' direction={'column'}>
            <Heading>Normal images</Heading>

            <Grid
                flex={1}
                columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
                rows={['max-content', '1fr']}
                gap={'size-100'}
                alignContent={'start'}
            >
                {mediaItemsToRender.map((mediaItem, index) => (
                    <DatasetItemContainer key={mediaItem?.id ?? index} mediaItem={mediaItem} />
                ))}
            </Grid>
        </Flex>
    );
};
