import { Flex, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitField } from '../rate-limit-field/rate-limit-field.component';
import { RosSinkConfig } from '../utils';

interface RosFieldsProps {
    defaultState: RosSinkConfig;
}

export const RosFields = ({ defaultState }: RosFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />

            <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />
            <TextField width='100%' label='Topic' name='topic' defaultValue={defaultState.topic} />

            <RateLimitField defaultValue={defaultState.rate_limit} />

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
