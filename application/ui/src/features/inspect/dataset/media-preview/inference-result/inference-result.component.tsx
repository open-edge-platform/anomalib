import { useState } from 'react';

import { Flex, Loading, View } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { useSpinDelay } from 'spin-delay';

import { useInference } from '../../../inference-provider.component';
import { MediaItem } from '../../types';
import { LabelScore } from './label-score.component';

import classes from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
}

export const InferenceResult = ({ selectedMediaItem }: InferenceResultProps) => {
    const [isVerticalImage, setIsVerticalImage] = useState(false);
    const { inferenceOpacity, isPending, inferenceResult } = useInference();

    const isLoadingInference = useSpinDelay(isPending, { delay: 300 });

    const handleInference = async (imageElement: HTMLImageElement) => {
        setIsVerticalImage(imageElement.clientHeight > imageElement.clientWidth);
    };

    return (
        <Flex height={'100%'} direction={'column'} alignItems={'center'} justifyContent={'center'}>
            <Flex height={'100%'} direction={'column'} alignItems={'baseline'} justifyContent={'center'}>
                {isLoadingInference && <Loading mode={'overlay'} />}
                {inferenceResult && <LabelScore label={inferenceResult.label} score={inferenceResult.score} />}

                <View position={'relative'} UNSAFE_style={{ height: isVerticalImage ? 'calc(100% - 28px)' : 'auto' }}>
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
