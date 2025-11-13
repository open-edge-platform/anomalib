import { createContext, ReactNode, use, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { components } from '@geti-inspect/api/spec';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { MediaItem } from './dataset/types';
import { useSelectedMediaItem } from './selected-media-item-provider.component';

type InferenceResult = components['schemas']['PredictionResponse'] | undefined;

interface InferenceContextProps {
    onInference: (media: MediaItem, modelId: string) => Promise<void>;
    inferenceResult: InferenceResult;
    isPending: boolean;
    selectedModelId: string | undefined;
    onSetSelectedModelId: (model: string | undefined) => void;
    inferenceOpacity: number;
    onInferenceOpacityChange: (opacity: number) => void;
}

const InferenceContext = createContext<InferenceContextProps | undefined>(undefined);

const downloadImageAsFile = async (media: MediaItem) => {
    const response = await fetch(`/api/projects/${media.project_id}/images/${media.id}/full`);

    const blob = await response.blob();

    return new File([blob], media.filename, { type: blob.type });
};

const useInferenceMutation = () => {
    const pipeline = usePipeline();
    const inferenceMutation = $api.useMutation('post', '/api/projects/{project_id}/models/{model_id}:predict');

    const handleInference = async (mediaItem: MediaItem, modelId: string) => {
        const file = await downloadImageAsFile(mediaItem);

        const formData = new FormData();
        formData.append('file', file);

        if (pipeline.data.inference_device) {
            formData.append('device', pipeline.data.inference_device);
        }

        inferenceMutation.mutate({
            // @ts-expect-error There is an incorrect type in OpenAPI
            body: formData,
            params: {
                path: {
                    project_id: mediaItem.project_id,
                    model_id: modelId,
                },
            },
        });
    };

    return {
        inferenceResult: inferenceMutation.data,
        onInference: handleInference,
        isPending: inferenceMutation.isPending,
    };
};

interface InferenceProviderProps {
    children: ReactNode;
}

export const InferenceProvider = ({ children }: InferenceProviderProps) => {
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const { selectedMediaItem } = useSelectedMediaItem();
    const [inferenceOpacity, setInferenceOpacity] = useState<number>(0.75);
    const { inferenceResult, onInference, isPending } = useInferenceMutation();

    const onSetSelectedModelId = (modelId: string | undefined) => {
        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id: modelId },
        });

        if (modelId && selectedMediaItem) {
            onInference(selectedMediaItem, modelId);
        }
    };
    return (
        <InferenceContext
            value={{
                onInference,
                isPending,
                inferenceResult,
                selectedModelId: pipeline.model?.id,
                inferenceOpacity,
                onSetSelectedModelId,
                onInferenceOpacityChange: setInferenceOpacity,
            }}
        >
            {children}
        </InferenceContext>
    );
};

export const useInference = () => {
    const context = use(InferenceContext);

    if (context === undefined) {
        throw new Error('useInference must be used within a InferenceProvider');
    }

    return context;
};
