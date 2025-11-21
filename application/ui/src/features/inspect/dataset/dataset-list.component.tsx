import { Flex, Grid, Heading, minmax, repeat } from '@geti/ui';
import isEmpty from 'lodash-es/isEmpty';

import { DatasetItemPlaceholder } from './dataset-item/dataset-item-placeholder.component';
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
        <Flex gap='size-200' direction={'column'}>
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
                        <DatasetItem key={mediaItem.id} mediaItem={mediaItem} />
                    )
                )}
            </Grid>
        </Flex>
    );
};
