import { useState } from 'react';

import { SchemaPredictionResponse } from '@geti-inspect/api/spec';
import { dimensionValue, Flex, View } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { isNonEmptyString } from 'src/features/inspect/utils';

import { MediaItem } from '../../types';
import { useInference } from '../providers/inference-opacity-provider.component';
import { LabelScore } from './label-score.component';

import classes from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
    inferenceResult: SchemaPredictionResponse | undefined;
}

export const InferenceResult = ({ selectedMediaItem, inferenceResult }: InferenceResultProps) => {
    const { inferenceOpacity } = useInference();
    const [isVerticalImage, setIsVerticalImage] = useState(false);

    const handleImageOrientation = (imageElement: HTMLImageElement) => {
        setIsVerticalImage(imageElement.clientHeight > imageElement.clientWidth);
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
                        onLoad={({ target }) => handleImageOrientation(target as HTMLImageElement)}
                    />

                    <AnimatePresence>
                        {isNonEmptyString(inferenceResult?.anomaly_map) && (
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
