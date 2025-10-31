import { createContext, ReactNode, use, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { components } from '@geti-inspect/api/spec';

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
    const inferenceMutation = $api.useMutation('post', '/api/projects/{project_id}/models/{model_id}:predict');

    const handleInference = async (mediaItem: MediaItem, modelId: string) => {
        const file = await downloadImageAsFile(mediaItem);

        const formData = new FormData();
        formData.append('file', file);

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
    const { inferenceResult, onInference, isPending } = useInferenceMutation();
    const [selectedModelId, setSelectedModelId] = useState<string | undefined>(undefined);
    const [inferenceOpacity, setInferenceOpacity] = useState<number>(0.75);

    const { selectedMediaItem } = useSelectedMediaItem();

    const onSetSelectedModelId = (modelId: string | undefined) => {
        setSelectedModelId(modelId);

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
                selectedModelId,
                onSetSelectedModelId,
                inferenceOpacity,
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
