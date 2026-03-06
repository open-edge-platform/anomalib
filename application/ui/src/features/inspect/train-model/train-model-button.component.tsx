import { Suspense } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Button, DialogTrigger } from '@geti/ui';

import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from '../dataset/utils';
import { TrainModelDialog } from './train-model-dialog.component';

const useIsTrainingButtonDisabled = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/images', {
        params: { path: { project_id: projectId } },
    });

    const uploadedNormalImages = data?.media.length ?? 0;

    return uploadedNormalImages < REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING;
};

export const TrainModelButton = () => {
    const isDisabled = useIsTrainingButtonDisabled();

    return (
        <Suspense
            fallback={
                <Button isDisabled isPending>
                    Train model
                </Button>
            }
        >
            <DialogTrigger type='modal'>
                <Button isDisabled={isDisabled}>Train model</Button>
                {(close) => <TrainModelDialog close={close} />}
            </DialogTrigger>
        </Suspense>
    );
};
