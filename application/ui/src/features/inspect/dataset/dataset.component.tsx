import { Suspense } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, Heading, Loading, View } from '@geti/ui';

import { TrainModelButton } from '../train-model/train-model-button.component';
import { DatasetList } from './dataset-list.component';
import { UploadImages } from './upload-images.component';

const useMediaItems = () => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/images', {
        params: { path: { project_id: projectId } },
    });

    return {
        mediaItems: data.media,
    };
};

export const Dataset = () => {
    const { mediaItems } = useMediaItems();
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
                        <DatasetList mediaItems={mediaItems} />
                    </Flex>
                </View>
            </Suspense>
        </Flex>
    );
};
