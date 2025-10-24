import { Suspense } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, FileTrigger, Flex, Heading, Loading, toast, View } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { TrainModelButton } from '../train-model/train-model-button.component';
import { DatasetList } from './dataset-list.component';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

const useMediaItems = () => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/images', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return {
        mediaItems: data.media,
    };
};

const UploadImages = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();

    const captureImageMutation = $api.useMutation('post', '/api/projects/{project_id}/capture');

    const handleAddMediaItem = async (files: File[]) => {
        const uploadPromises = files.map((file) => {
            const formData = new FormData();
            formData.append('file', file);

            return captureImageMutation.mutateAsync({
                params: { path: { project_id: projectId } },
                // @ts-expect-error There is an incorrect type in OpenAPI
                body: formData,
            });
        });

        const promises = await Promise.allSettled(uploadPromises);

        const succeeded = promises.filter((result) => result.status === 'fulfilled').length;
        const failed = promises.filter((result) => result.status === 'rejected').length;

        const imagesOptions = $api.queryOptions('get', '/api/projects/{project_id}/images', {
            params: { path: { project_id: projectId } },
        });
        await queryClient.invalidateQueries({ queryKey: imagesOptions.queryKey });
        const images = await queryClient.ensureQueryData(imagesOptions);

        if (images.media.length >= REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING) {
            toast({
                title: 'Train',
                type: 'info',
                message: `You can start model training now with your collected dataset.`,
                duration: Infinity,
                actionButtons: [<TrainModelButton key='train' />],
                position: 'bottom-left',
            });
            return;
        }

        if (failed === 0) {
            toast({ type: 'success', message: `Uploaded ${succeeded} item(s)` });
        } else if (succeeded === 0) {
            toast({ type: 'error', message: `Failed to upload ${failed} item(s)` });
        } else {
            toast({
                type: 'warning',
                message: `Uploaded ${succeeded} item(s), ${failed} failed`,
            });
        }
    };

    const captureImages = (files: FileList | null) => {
        if (files === null) return;

        handleAddMediaItem(Array.from(files));
    };

    return (
        <FileTrigger allowsMultiple onSelect={captureImages}>
            <Button variant='secondary'>Upload images</Button>
        </FileTrigger>
    );
};

const DatasetContent = () => {
    const { mediaItems } = useMediaItems();

    return <DatasetList mediaItems={mediaItems} />;
};

export const Dataset = () => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading margin={0}>
                <Flex justifyContent={'space-between'}>
                    Dataset
                    <Flex gap='size-200'>
                        <UploadImages />
                        <TrainModelButton />
                    </Flex>
                </Flex>
            </Heading>
            <Suspense fallback={<Loading mode={'inline'} />}>
                <View flex={1} padding={'size-300'}>
                    <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                        <DatasetContent />
                    </Flex>
                </View>
            </Suspense>
        </Flex>
    );
};
