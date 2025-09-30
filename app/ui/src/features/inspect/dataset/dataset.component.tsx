import { Suspense } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import {
    Button,
    Content,
    Divider,
    FileTrigger,
    Flex,
    Grid,
    Heading,
    InlineAlert,
    Loading,
    minmax,
    repeat,
    View,
} from '@geti/ui';

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

const REQUIRED_NUMBER_OF_NORMAL_IMAGES = 20;

const useMediaItems = () => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/images', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    if (data.media.length >= REQUIRED_NUMBER_OF_NORMAL_IMAGES) {
        return {
            mediaItems: data.media,
        };
    }

    const mediaItems = Array.from({ length: REQUIRED_NUMBER_OF_NORMAL_IMAGES }).map((_, index) =>
        index <= data.media.length - 1 ? data.media[index] : undefined
    );

    return { mediaItems };
};

const DatasetItemsList = () => {
    const { mediaItems } = useMediaItems();

    return (
        <Grid
            flex={1}
            columns={repeat('auto-fill', minmax('size-1600', '1fr'))}
            gap={'size-100'}
            alignContent={'start'}
        >
            {mediaItems.map((mediaItem, index) => (
                <DatasetItem key={mediaItem?.id ?? index} mediaItem={mediaItem} />
            ))}
        </Grid>
    );
};

const UploadImages = () => {
    const { projectId } = useProjectIdentifier();

    const captureImageMutation = $api.useMutation('post', '/api/projects/{project_id}/capture');

    const captureImage = (file: File) => {
        const formData = new FormData();
        formData.append('file', file);

        captureImageMutation.mutate({
            body: {
                // @ts-expect-error Incorrect OpenAPI types
                file: formData,
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const captureImages = (files: FileList) => {
        Array.from(files).forEach((file) => captureImage(file));
    };

    return (
        <FileTrigger
            onSelect={(files) => {
                files?.item();
                console.log(files);
            }}
        >
            <Button>Upload images</Button>
        </FileTrigger>
    );
};

export const Dataset = () => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading margin={0}>
                Dataset <UploadImages />
            </Heading>
            <Suspense fallback={<Loading mode={'inline'} />}>
                <View flex={1} padding={'size-300'}>
                    <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                        <NotEnoughNormalImagesToTrain />

                        <Divider size={'S'} />

                        <DatasetItemsList />
                    </Flex>
                </View>
            </Suspense>
        </Flex>
    );
};
