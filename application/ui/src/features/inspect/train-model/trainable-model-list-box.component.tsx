import { Badge } from '@adobe/react-spectrum';
import { $api } from '@geti-inspect/api';
import { SchemaModelFamily as ModelFamily, SchemaTrainingTime as TrainingTime } from '@geti-inspect/api/spec';
import { Flex, Grid, Heading, minmax, Radio, repeat, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { Coffee, Cycle, Walk } from 'src/assets/icons';

import classes from './train-model.module.scss';

const useTrainableModels = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/trainable-models', undefined, {
        staleTime: Infinity,
        gcTime: Infinity,
    });

    return data.trainable_models.map((model) => ({
        id: model.name, // use the name as id
        name: model.name,
        training_time: model.training_time,
        model_family: model.model_family,
        recommended: model.recommended,
        license: model.license,
    }));
};

interface TrainableModel {
    id: string;
    name: string;
    training_time: TrainingTime;
    model_family: ModelFamily[];
    recommended: boolean;
    license: string;
}

const MODEL_FAMILY_COLORS: Record<ModelFamily, string> = {
    patch_based: classes.modelFamilyPatchBased,
    memory_bank: classes.modelFamilyMemoryBank,
    student_teacher: classes.modelFamilyStudentTeacher,
    reconstruction_based: classes.modelFamilyReconstruction,
    distribution_map: classes.modelFamilyDistributionMap,
};

/**
 * Get the name of a model family.
 * @param modelFamily - Name in snake_case.
 * @returns Name in title case for example 'Patch Based' from 'patch_based'.
 */
const getModelFamilyName = (modelFamily: ModelFamily) => {
    let parts = modelFamily.split('_');
    return parts.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
};

const TrainingTimeIcon = ({ trainingTime }: { trainingTime: TrainingTime }) => {
    switch (trainingTime) {
        case 'coffee':
            return <Coffee />;
        case 'walk':
            return <Walk />;
        case 'cycle':
            return <Cycle />;
        default:
            return null;
    }
};

interface ModelProps {
    model: TrainableModel;
    isSelected?: boolean;
}

const Model = ({ model, isSelected = false }: ModelProps) => {
    const { name, id, training_time, model_family, recommended, license } = model;

    return (
        <label
            htmlFor={`select-model-${id}`}
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
                <Flex alignItems={'center'} justifyContent={'space-between'} gap={'size-50'} marginBottom={'size-50'}>
                    <Radio value={id} aria-label={name} id={`select-model-${id}`}>
                        <Heading UNSAFE_className={clsx({ [classes.selected]: isSelected })}>{name}</Heading>
                    </Radio>
                    <Badge variant='neutral' UNSAFE_className={classes.licenseBadge}>
                        {license}
                    </Badge>
                </Flex>
                {recommended && (
                    <Flex>
                        <Badge variant='info' UNSAFE_className={classes.badge}>
                            ★ Recommended
                        </Badge>
                    </Flex>
                )}
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
                <Flex direction={'column'} gap={'size-200'} justifyContent={'space-between'} height={'100%'}>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        <Text>Training time:</Text>
                        <View UNSAFE_className={classes.trainingTimeIcon}>
                            <TrainingTimeIcon trainingTime={training_time} />
                        </View>
                    </Flex>
                    <Flex alignItems={'center'} gap={'size-100'} wrap marginTop={'size-200'}>
                        {model_family.map((family) => (
                            <Badge
                                key={family}
                                variant='neutral'
                                UNSAFE_className={clsx(classes.badge, MODEL_FAMILY_COLORS[family])}
                            >
                                {getModelFamilyName(family)}
                            </Badge>
                        ))}
                    </Flex>
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

    return (
        <Grid columns={repeat('auto-fit', minmax('size-3400', '1fr'))} gap={'size-250'}>
            {trainableModels.map((model) => {
                const isSelected = selectedModelTemplateId === model.id;

                return <Model key={model.id} model={model} isSelected={isSelected} />;
            })}
        </Grid>
    );
};
