import { Content, Divider, Flex, Grid, Heading, InlineAlert, minmax, repeat, View } from '@geti/ui';

import { DatasetItem } from './dataset-item/dataset-item.component';

const NotEnoughNormalImagesToTrain = () => {
    // TODO: This should change dynamically when user provides more normal images
    const requiredNumberOfNormalImages = 20;

    return (
        <InlineAlert variant='info'>
            <Heading>{requiredNumberOfNormalImages} images required</Heading>
            <Content>
                Capture {requiredNumberOfNormalImages} images of normal cases. They help the model learn what is
                standard, so it can better detect anomalies.
            </Content>
        </InlineAlert>
    );
};

const DatasetItemsList = () => {
    const mediaItems = Array.from({ length: 20 }).map((_, index) => ({
        id: index,
        mediaItem: undefined,
    }));

    return (
        <Grid
            flex={1}
            columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
            gap={'size-100'}
            alignContent={'start'}
        >
            {mediaItems.map(({ id, mediaItem }) => (
                <DatasetItem key={id} mediaItem={mediaItem} />
            ))}
        </Grid>
    );
};

export const Dataset = () => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading margin={0}>Dataset</Heading>
            <View flex={1} padding={'size-300'}>
                <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                    <NotEnoughNormalImagesToTrain />

                    <Divider size={'S'} />

                    <DatasetItemsList />
                </Flex>
            </View>
        </Flex>
    );
};
