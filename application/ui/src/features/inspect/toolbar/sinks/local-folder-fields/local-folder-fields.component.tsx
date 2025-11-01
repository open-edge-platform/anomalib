import { Flex, NumberField, TextField } from '@geti/ui';

import { ReactComponent as FolderIcon } from '../../../../../assets/icons/folder.svg';
import { OutputFormats } from '../output-formats.component';
import { LocalFolderSinkConfig } from '../utils';

import classes from './local-folder-fields.module.scss';

interface LocalFolderFieldsProps {
    state: LocalFolderSinkConfig;
}

export const LocalFolderFields = ({ state }: LocalFolderFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={state.id} />
            <Flex direction={'row'} gap='size-200'>
                <TextField label='Name' name='name' defaultValue={state.name} />
                <NumberField
                    label='Rate Limit'
                    name='rate_limit'
                    minValue={0}
                    step={0.1}
                    defaultValue={state.rate_limit ?? undefined}
                />
            </Flex>

            <Flex direction='row' gap='size-200'>
                <TextField width={'100%'} label='Folder Path' name='folder_path' defaultValue={state.folder_path} />

                <Flex
                    alignSelf={'end'}
                    height={'size-400'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    UNSAFE_className={classes.folderIcon}
                >
                    <FolderIcon />
                </Flex>
            </Flex>

            <OutputFormats config={state.output_formats} />
        </Flex>
    );
};
