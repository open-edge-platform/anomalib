import { Flex, Grid, Loading, Text } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { useSpinDelay } from 'spin-delay';

import { MediaItem } from '../../dataset/types';
import { useInference } from '../../inference-provider.component';

import styles from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
}

interface LabelProps {
    label: string;
    score: number;
}

const LabelScore = ({ label, score }: LabelProps) => {
    const formatter = new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 0,
        style: 'percent',
    });

    return (
        <Flex
            gridArea={'label'}
            UNSAFE_className={clsx(styles.label, {
                [styles.labelNormal]: label.toLowerCase() === 'normal',
                [styles.labelAnomalous]: label.toLowerCase() === 'anomalous',
            })}
            gap={'size-50'}
        >
            <Text>{label}</Text>
            <Text>{formatter.format(score)}</Text>
        </Flex>
    );
};

export const InferenceResult = ({ selectedMediaItem }: InferenceResultProps) => {
    const { isPending, inferenceResult, inferenceOpacity } = useInference();
    const isLoadingInference = useSpinDelay(isPending, { delay: 300 });

    return (
        <Grid
            gridArea={'canvas'}
            position={'relative'}
            justifyContent={'center'}
            rows={['max-content', 'minmax(0, 1fr)']}
            columns={['max-content', 'max-content']}
            areas={['label .', 'inference-result inference-result']}
            UNSAFE_className={styles.canvasContainer}
            UNSAFE_style={{ overflow: 'hidden' }}
        >
            <img
                src={`/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`}
                alt={selectedMediaItem.filename}
                className={clsx(styles.img, styles.inferencedImage)}
            />
            <AnimatePresence>
                {inferenceResult !== undefined && (
                    <>
                        <LabelScore label={inferenceResult.label} score={inferenceResult.score} />
                        <motion.img
                            exit={{ opacity: 0 }}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: inferenceOpacity }}
                            src={`data:image/png;base64,${inferenceResult.anomaly_map}`}
                            alt={`${selectedMediaItem.filename} inference`}
                            className={clsx(styles.img, styles.inferencedImage)}
                            style={{ opacity: inferenceOpacity }}
                        />
                    </>
                )}
            </AnimatePresence>
            {isLoadingInference && <Loading mode={'overlay'} />}
        </Grid>
    );
};
