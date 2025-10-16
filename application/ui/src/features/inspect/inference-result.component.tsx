import { Flex, Text } from '@adobe/react-spectrum';
import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Grid, Heading, Loading } from '@geti/ui';
import { clsx } from 'clsx';
import { AnimatePresence, motion } from 'motion/react';
import { useSpinDelay } from 'spin-delay';

import { useInference } from './inference-provider.component';
import { useSelectedMediaItem } from './selected-media-item-provider.component';

import styles from './inference.module.scss';

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

const useIsInferenceAvailable = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data?.models.length !== 0;
};

export const InferenceResult = () => {
    const { selectedMediaItem } = useSelectedMediaItem();
    const { isPending, inferenceResult, inferenceOpacity } = useInference();
    const isInferenceAvailable = useIsInferenceAvailable();
    const isLoadingInference = useSpinDelay(isPending, { delay: 300 });

    if (!isInferenceAvailable && selectedMediaItem === undefined) {
        return (
            <Grid
                gridArea={'canvas'}
                UNSAFE_className={styles.canvasContainer}
                justifyContent={'center'}
                alignContent={'center'}
            >
                <Heading>No trained models available. Please train a model to start inference.</Heading>
            </Grid>
        );
    }

    if (selectedMediaItem === undefined) {
        return (
            <Grid
                gridArea={'canvas'}
                UNSAFE_className={styles.canvasContainer}
                justifyContent={'center'}
                alignContent={'center'}
            >
                <Heading>Select an image to start inference and receive predictions.</Heading>
            </Grid>
        );
    }

    const mediaUrl = `/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`;

    return (
        <Grid
            gridArea={'canvas'}
            columns={['max-content', 'max-content']}
            rows={['max-content', 'minmax(0, 1fr)']}
            areas={['label .', 'inference-result inference-result']}
            justifyContent={'center'}
            UNSAFE_className={styles.canvasContainer}
            UNSAFE_style={{
                overflow: 'hidden',
            }}
            position={'relative'}
        >
            <img src={mediaUrl} alt={selectedMediaItem.filename} className={clsx(styles.img, styles.inferencedImage)} />
            <AnimatePresence>
                {inferenceResult !== undefined && (
                    <>
                        <LabelScore label={inferenceResult.label} score={inferenceResult.score} />
                        <motion.img
                            initial={{ opacity: 0 }}
                            exit={{ opacity: 0 }}
                            animate={{ opacity: inferenceOpacity }}
                            src={`data:image/png;base64,${inferenceResult.anomaly_map}`}
                            alt={`${selectedMediaItem.filename} inference`}
                            className={clsx(styles.img, styles.inferencedImage)}
                            style={{
                                opacity: inferenceOpacity,
                            }}
                        />
                    </>
                )}
            </AnimatePresence>
            {isLoadingInference && <Loading mode={'overlay'} />}
        </Grid>
    );
};
