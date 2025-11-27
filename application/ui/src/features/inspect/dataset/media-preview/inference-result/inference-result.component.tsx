import { useEffect, useState } from 'react';

import { usePipeline } from '@geti-inspect/hooks';
import { dimensionValue, Flex, View } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';

import { useInference } from '../../../inference-provider.component';
import { MediaItem } from '../../types';
import { LabelScore } from './label-score.component';

import classes from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
}

export const InferenceResult = ({ selectedMediaItem }: InferenceResultProps) => {
    const { data: pipeline } = usePipeline();
    const [isVerticalImage, setIsVerticalImage] = useState(false);
    const { inferenceOpacity, inferenceResult, onInference, resetInference } = useInference();
    const selectedModelId = pipeline?.model?.id;

    useEffect(() => {
        return () => {
            resetInference();
        };
    }, [resetInference]);

    const handleInference = async (imageElement: HTMLImageElement) => {
        setIsVerticalImage(imageElement.clientHeight > imageElement.clientWidth);
        selectedModelId && onInference(selectedMediaItem, selectedModelId);
    };

    return (
        <Flex height={'100%'} direction={'column'} alignItems={'center'} justifyContent={'center'}>
            <Flex height={'100%'} direction={'column'} alignItems={'baseline'} justifyContent={'center'}>
                <View height={'size-350'}>
                    {inferenceResult && <LabelScore label={inferenceResult.label} score={inferenceResult.score} />}
                </View>

                <View
                    position={'relative'}
                    UNSAFE_style={{ height: isVerticalImage ? `calc(100% - ${dimensionValue('size-350')})` : 'auto' }}
                >
                    <img
                        alt={selectedMediaItem.filename}
                        className={clsx(classes.img, { [classes.verticalImg]: isVerticalImage })}
                        src={`/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`}
                        onLoad={({ target }) => handleInference(target as HTMLImageElement)}
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
                                    className={clsx(classes.inferenceImage)}
                                    style={{ opacity: inferenceOpacity }}
                                />
                            </>
                        )}
                    </AnimatePresence>
                </View>
            </Flex>
        </Flex>
    );
};
