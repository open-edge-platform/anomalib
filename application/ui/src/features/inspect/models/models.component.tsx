import { Suspense } from 'react';

import { Flex, Heading, Loading } from '@geti/ui';

import { TrainModelButton } from '../train-model/train-model-button.component';
import { InferenceDevices } from './inference-devices.component';
import { ModelsView } from './models-view.component';

export const Models = () => {
    return (
        <Flex direction={'column'} height={'100%'} gap={'size-250'}>
            <Heading margin={0}>
                <Flex justifyContent={'space-between'}>
                    Models
                    <Flex gap='size-200'>
                        <TrainModelButton />
                    </Flex>
                </Flex>
            </Heading>
            <Suspense fallback={<Loading mode={'inline'} />}>
                <>
                    <InferenceDevices />
                    <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                        <ModelsView />
                    </Flex>
                </>
            </Suspense>
        </Flex>
    );
};
