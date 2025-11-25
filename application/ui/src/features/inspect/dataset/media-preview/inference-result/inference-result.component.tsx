import { $api } from '@geti-inspect/api';
import { usePipeline } from '@geti-inspect/hooks';
import { Loading, View } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { useSpinDelay } from 'spin-delay';

import { useInference } from '../../../inference-provider.component';
import { MediaItem } from '../../types';
import { LabelScore } from './label-score.component';
import { downloadImageAsFile } from './util';

import styles from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
}

export const InferenceResult = ({ selectedMediaItem }: InferenceResultProps) => {
    const pipeline = usePipeline();
    const { inferenceOpacity, selectedModelId } = useInference();
    const {
        isPending,
        data: inferenceResult,
        mutate: inferenceMutation,
    } = $api.useMutation('post', '/api/projects/{project_id}/models/{model_id}:predict');

    const isLoadingInference = useSpinDelay(isPending, { delay: 300 });

    const handleInference = async (mediaItem: MediaItem, modelId?: string) => {
        if (!modelId) {
            return;
        }

        const file = await downloadImageAsFile(mediaItem);

        const formData = new FormData();
        formData.append('file', file);

        if (pipeline.data.inference_device) {
            formData.append('device', pipeline.data.inference_device);
        }

        inferenceMutation({
            // @ts-expect-error There is an incorrect type in OpenAPI
            body: formData,
            params: { path: { project_id: mediaItem.project_id, model_id: modelId } },
        });
    };

    return (
        <View width={'100%'} height={'100%'}>
            {isLoadingInference && <Loading mode={'overlay'} />}
            {inferenceResult && <LabelScore label={inferenceResult.label} score={inferenceResult.score} />}

            <View position={'relative'}>
                <img
                    className={clsx(styles.img)}
                    alt={selectedMediaItem.filename}
                    src={`/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`}
                    onLoad={() => handleInference(selectedMediaItem, selectedModelId!)}
                />

                <AnimatePresence>
                    {inferenceResult !== undefined && (
                        <>
                            <motion.img
                                exit={{ opacity: 0 }}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: inferenceOpacity }}
                                src={`data:image/png;base64,${inferenceResult.anomaly_map}`}
                                alt={`${selectedMediaItem.filename} inference`}
                                className={clsx(styles.inferenceImage)}
                                style={{ opacity: inferenceOpacity }}
                            />
                        </>
                    )}
                </AnimatePresence>
            </View>
        </View>
    );
};
