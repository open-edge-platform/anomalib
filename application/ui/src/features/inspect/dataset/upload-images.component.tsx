// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, FileTrigger, toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { useUploadStatus } from '../footer/status-bar/adapters/use-upload-status';
import { TrainModelButton } from '../train-model/train-model-button.component';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

export const UploadImages = () => {
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();
    const { startUpload, incrementProgress } = useUploadStatus();

    const captureImageMutation = $api.useMutation('post', '/api/projects/{project_id}/images', {
        onSuccess: () => incrementProgress(true),
        onError: () => incrementProgress(false),
    });

    const handleAddMediaItem = async (files: File[]) => {
        startUpload(files.length);

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
