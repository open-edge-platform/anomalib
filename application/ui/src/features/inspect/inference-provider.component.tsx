import { createContext, ReactNode, use, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { components } from '@geti-inspect/api/spec';
import { toast } from 'packages/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { MediaItem } from './dataset/types';

type InferenceResult = components['schemas']['PredictionResponse'] | undefined;

interface InferenceContextProps {
    inferenceResult: InferenceResult;
    isPending: boolean;
    inferenceOpacity: number;
    resetInference: () => void;
    onInference: (media: MediaItem, modelId: string) => Promise<void>;
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
    const inferenceMutation = $api.useMutation('post', '/api/projects/{project_id}/models/{model_id}:predict', {
        onError: () => {
            toast({ type: 'error', message: 'Unable to process the image. Please try again.' });
        },
    });

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
        onInference: handleInference,
        isPending: inferenceMutation.isPending,
        resetInference: inferenceMutation.reset,
        inferenceResult: inferenceMutation.data,
    };
};

interface InferenceProviderProps {
    children: ReactNode;
}

export const InferenceProvider = ({ children }: InferenceProviderProps) => {
    const [inferenceOpacity, setInferenceOpacity] = useState<number>(0.75);
    const { inferenceResult, isPending, onInference, resetInference } = useInferenceMutation();

    return (
        <InferenceContext
            value={{
                onInference,
                isPending,
                inferenceResult,
                inferenceOpacity,
                resetInference,
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
