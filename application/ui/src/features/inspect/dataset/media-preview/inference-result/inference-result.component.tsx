import { useEffect, useRef, useState } from 'react';

import { SchemaPredictionResponse } from '@geti-inspect/api/spec';
import { DimensionValue, Responsive, View } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { isNonEmptyString } from 'src/features/inspect/utils';

import { MediaItem } from '../../types';
import { useInference } from '../providers/inference-opacity-provider.component';
import { LabelScore } from './label-score.component';
import { getImageDimensions } from './util';

import classes from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
    inferenceResult: SchemaPredictionResponse | undefined;
}

const labelHeight: Responsive<DimensionValue> = 'size-350';

export const InferenceResult = ({ selectedMediaItem, inferenceResult }: InferenceResultProps) => {
    const imageRef = useRef(null);
    const { inferenceOpacity } = useInference();
    const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0, left: 0, top: 0 });

    const handleImageLoaded = (imageElement: HTMLImageElement) => {
        setImageDimensions(getImageDimensions(imageElement));
    };

    useEffect(() => {
        if (!imageRef.current) return;

        const observer = new ResizeObserver(([entry]) => {
            handleImageLoaded(entry.target as HTMLImageElement);
        });

        observer.observe(imageRef.current);
        return () => observer.disconnect();
    }, []);

    return (
        <View height={'100%'} paddingTop={labelHeight}>
            {inferenceResult && (
                <View
                    height={labelHeight}
                    position={'absolute'}
                    maxWidth={'size-1600'}
                    top={imageDimensions.top}
                    left={imageDimensions.left}
                >
                    <LabelScore label={inferenceResult.label} score={inferenceResult.score} />
                </View>
            )}

            <View width={'100%'} height={'100%'} position={'relative'}>
                <img
                    ref={imageRef}
                    alt={selectedMediaItem.filename}
                    className={clsx(classes.img)}
                    src={`/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`}
                    onLoad={({ target }) => handleImageLoaded(target as HTMLImageElement)}
                />

                <AnimatePresence>
                    {isNonEmptyString(inferenceResult?.anomaly_map) && (
                        <motion.img
                            exit={{ opacity: 0 }}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: inferenceOpacity }}
                            className={clsx(classes.inferenceImage)}
                            style={{ ...imageDimensions }}
                            src={`data:image/png;base64,${inferenceResult.anomaly_map}`}
                            alt={`${selectedMediaItem.filename} inference`}
                        />
                    )}
                </AnimatePresence>
            </View>
        </View>
    );
};
