import { $api } from '@geti-inspect/api';
import { Flex, Grid, Heading, minmax, Radio, repeat, View } from '@geti/ui';
import { clsx } from 'clsx';

import classes from './train-model.module.scss';

const useTrainableModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/trainable-models', undefined, {
        staleTime: Infinity,
        gcTime: Infinity,
    });

    return data.trainable_models.map((model) => ({ id: model, name: model }));
};

type Ratings = 'LOW' | 'MEDIUM' | 'HIGH';

const RateColorPalette = {
    LOW: 'var(--energy-blue-tint2)',
    MEDIUM: 'var(--energy-blue-tint1)',
    HIGH: 'var(--energy-blue)',
    EMPTY: 'var(--spectrum-global-color-gray-500)',
};

const RateColors = {
    LOW: [RateColorPalette.LOW, RateColorPalette.EMPTY, RateColorPalette.EMPTY],
    MEDIUM: [RateColorPalette.LOW, RateColorPalette.MEDIUM, RateColorPalette.EMPTY],
    HIGH: [RateColorPalette.LOW, RateColorPalette.MEDIUM, RateColorPalette.HIGH],
};
const RATE_LABELS = Object.keys(RateColors);

interface AttributeRatingProps {
    name: string;
    rating: Ratings;
}

const AttributeRating = ({ name, rating }: AttributeRatingProps) => {
    return (
        <div aria-label={`Attribute rating for ${name} is ${rating}`} style={{ height: '100%' }}>
            <Flex direction={'column'} gap={'size-100'} justifyContent={'space-between'} height={'100%'}>
                <Heading margin={0} UNSAFE_className={classes.attributeRatingTitle}>
                    {name}
                </Heading>
                <Flex alignItems={'center'} gap={'size-100'}>
                    {RateColors[rating].map((color, idx) => (
                        <View
                            key={`rate-${RATE_LABELS[idx]}`}
                            UNSAFE_className={classes.rate}
                            UNSAFE_style={{
                                backgroundColor: color,
                            }}
                        />
                    ))}
                </Flex>
            </Flex>
        </div>
    );
};

enum PerformanceCategory {
    OTHER = 'other',
    SPEED = 'speed',
    BALANCE = 'balance',
    ACCURACY = 'accuracy',
}

type SupportedAlgorithmStatsValues = 1 | 2 | 3;

interface SupportedAlgorithm {
    name: string;
    modelTemplateId: string;
    performanceCategory: PerformanceCategory;
    performanceRatings: {
        accuracy: SupportedAlgorithmStatsValues;
        inferenceSpeed: SupportedAlgorithmStatsValues;
        trainingTime: SupportedAlgorithmStatsValues;
    };
    license: string;
}

interface TemplateRatingProps {
    ratings: {
        inferenceSpeed: Ratings;
        trainingTime: Ratings;
        accuracy: Ratings;
    };
}

const TemplateRating = ({ ratings }: TemplateRatingProps) => {
    return (
        <Grid columns={repeat(3, '1fr')} justifyContent={'space-evenly'} gap={'size-250'}>
            <AttributeRating name={'Inference speed'} rating={ratings.inferenceSpeed} />
            <AttributeRating name={'Training time'} rating={ratings.trainingTime} />
            <AttributeRating name={'Accuracy'} rating={ratings.accuracy} />
        </Grid>
    );
};

type PerformanceRating = SupportedAlgorithm['performanceRatings'][keyof SupportedAlgorithm['performanceRatings']];

const RATING_MAP: Record<PerformanceRating, Ratings> = {
    1: 'LOW',
    2: 'MEDIUM',
    3: 'HIGH',
};

interface ModelProps {
    algorithm: SupportedAlgorithm;
    isSelected?: boolean;
}

const Model = ({ algorithm, isSelected = false }: ModelProps) => {
    const { name, modelTemplateId, performanceRatings } = algorithm;

    return (
        <label
            htmlFor={`select-model-${algorithm.modelTemplateId}`}
            aria-label={isSelected ? 'Selected card' : 'Not selected card'}
            className={[classes.selectableCard, isSelected ? classes.selectableCardSelected : ''].join(' ')}
        >
            <View
                position={'relative'}
                paddingX={'size-175'}
                paddingY={'size-125'}
                borderTopWidth={'thin'}
                borderTopEndRadius={'regular'}
                borderTopStartRadius={'regular'}
                borderTopColor={'gray-200'}
                backgroundColor={'gray-200'}
                UNSAFE_className={isSelected ? classes.selectedHeader : ''}
            >
                <Flex alignItems={'center'} gap={'size-50'} marginBottom='size-50'>
                    <Radio value={modelTemplateId} aria-label={name} id={`select-model-${algorithm.modelTemplateId}`}>
                        <Heading UNSAFE_className={clsx({ [classes.selected]: isSelected })}>{name}</Heading>
                    </Radio>
                </Flex>
            </View>
            <View
                flex={1}
                paddingX={'size-250'}
                paddingY={'size-225'}
                borderBottomWidth={'thin'}
                borderBottomEndRadius={'regular'}
                borderBottomStartRadius={'regular'}
                borderBottomColor={'gray-100'}
                minHeight={'size-1000'}
                UNSAFE_className={[
                    classes.selectableCardDescription,
                    isSelected ? classes.selectedDescription : '',
                ].join(' ')}
            >
                <Flex direction={'column'} gap={'size-200'}>
                    <TemplateRating
                        ratings={{
                            accuracy: RATING_MAP[performanceRatings.accuracy],
                            trainingTime: RATING_MAP[performanceRatings.trainingTime],
                            inferenceSpeed: RATING_MAP[performanceRatings.inferenceSpeed],
                        }}
                    />
                </Flex>
            </View>
        </label>
    );
};

interface ModelTypesListProps {
    selectedModelTemplateId: string | null;
}

export const TrainableModelListBox = ({ selectedModelTemplateId }: ModelTypesListProps) => {
    const trainableModels = useTrainableModels();

    // NOTE: we will need to update the trainable models endpoint to return more info
    const models = trainableModels.map((model) => {
        return {
            modelTemplateId: model.id,
            name: model.name,
            license: 'Apache 2.0',
            performanceRatings: {
                accuracy: 1,
                inferenceSpeed: 1,
                trainingTime: 1,
            },
            performanceCategory: PerformanceCategory.OTHER,
        } satisfies SupportedAlgorithm;
    });

    return (
        <Grid columns={repeat('auto-fit', minmax('size-3400', '1fr'))} gap={'size-250'}>
            {models.map((algorithm) => {
                const isSelected = selectedModelTemplateId === algorithm.modelTemplateId;

                return <Model key={algorithm.modelTemplateId} algorithm={algorithm} isSelected={isSelected} />;
            })}
        </Grid>
    );
};
