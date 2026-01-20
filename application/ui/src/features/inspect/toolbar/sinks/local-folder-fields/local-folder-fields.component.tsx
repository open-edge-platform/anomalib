import { Flex, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitField } from '../rate-limit-field/rate-limit-field.component';
import { LocalFolderSinkConfig } from '../utils';

interface LocalFolderFieldsProps {
    defaultState: LocalFolderSinkConfig;
}

export const LocalFolderFields = ({ defaultState }: LocalFolderFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />

            <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />

            <Flex direction='row' gap='size-200'>
                <TextField
                    isRequired
                    width={'100%'}
                    label={'Folder Path'}
                    name={'folder_path'}
                    defaultValue={defaultState.folder_path}
                />
            </Flex>

            <RateLimitField defaultValue={defaultState.rate_limit} />

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
