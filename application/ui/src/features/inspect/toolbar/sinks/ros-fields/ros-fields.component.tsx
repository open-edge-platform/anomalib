import { Flex, NumberField, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RosSinkConfig } from '../utils';

interface RosFieldsProps {
    defaultState: RosSinkConfig;
}

export const RosFields = ({ defaultState }: RosFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />

            <Flex direction='row' gap='size-200' justifyContent='space-between'>
                <TextField flex='1' label='Name' name='name' defaultValue={defaultState.name} />
                <NumberField
                    label='Rate Limit'
                    name='rate_limit'
                    minValue={0}
                    step={0.1}
                    defaultValue={defaultState.rate_limit ?? undefined}
                />
            </Flex>

            <TextField width='100%' label='Topic' name='topic' defaultValue={defaultState.topic} />

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
