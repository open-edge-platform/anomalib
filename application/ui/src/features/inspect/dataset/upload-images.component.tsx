import { useRef } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, FileTrigger, toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { useUploadStatus } from '../footer/adapters';
import { TrainModelButton } from '../train-model/train-model-button.component';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

export const UploadImages = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();
    const { startUpload, updateProgress, completeUpload } = useUploadStatus();

    // Track progress across parallel uploads
    const progressRef = useRef({ completed: 0, failed: 0, total: 0 });

    const captureImageMutation = $api.useMutation('post', '/api/projects/{project_id}/images', {
        onSuccess: () => {
            progressRef.current.completed++;
            updateProgress({
                completed: progressRef.current.completed + progressRef.current.failed,
                total: progressRef.current.total,
                failed: progressRef.current.failed,
            });
        },
        onError: () => {
            progressRef.current.failed++;
            updateProgress({
                completed: progressRef.current.completed + progressRef.current.failed,
                total: progressRef.current.total,
                failed: progressRef.current.failed,
            });
        },
    });

    const handleAddMediaItem = async (files: File[]) => {
        const total = files.length;

        progressRef.current = { completed: 0, failed: 0, total };
        startUpload(total);

        const uploadPromises = files.map((file) => {
            const formData = new FormData();
            formData.append('file', file);

            return captureImageMutation.mutateAsync({
                params: { path: { project_id: projectId } },
                // @ts-expect-error There is an incorrect type in OpenAPI
                body: formData,
            });
        });

        await Promise.allSettled(uploadPromises);

        const { failed } = progressRef.current;
        completeUpload(failed === 0, failed);

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
