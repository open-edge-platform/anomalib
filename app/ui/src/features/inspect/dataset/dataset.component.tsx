import { Suspense } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, Divider, FileTrigger, Flex, Heading, Loading, View } from '@geti/ui';

import { DatasetList } from './dataset-list.component';
import { DatasetStatusPanel } from './dataset-status-panel.component';

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

    const captureImageMutation = $api.useMutation('post', '/api/projects/{project_id}/capture');

    const captureImage = (file: File) => {
        const formData = new FormData();
        formData.append('file', file);

        captureImageMutation.mutate({
            // @ts-expect-error There is an incorrect type in OpenAPI
            body: formData,
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const captureImages = (files: FileList | null) => {
        if (files === null) return;

        Array.from(files).forEach((file) => captureImage(file));
    };

    return (
        <FileTrigger allowsMultiple onSelect={captureImages}>
            <Button>Upload images</Button>
        </FileTrigger>
    );
};

const DatasetContent = () => {
    const { mediaItems } = useMediaItems();

    return (
        <>
            <DatasetStatusPanel mediaItemsCount={mediaItems.length} />

            <Divider size={'S'} />

            <DatasetList mediaItems={mediaItems} />
        </>
    );
};

export const Dataset = () => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading margin={0}>
                <Flex justifyContent={'space-between'}>
                    Dataset <UploadImages />
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
