// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Switch, TextField } from '@geti/ui';

import { ReactComponent as Folder } from '../../../../../assets/icons/folder.svg';
import { ImagesFolderSourceConfig } from '../util';

import classes from './image-folder-fields.module.scss';

type ImageFolderFieldsProps = {
    state: ImagesFolderSourceConfig;
};

export const ImageFolderFields = ({ state }: ImageFolderFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={state.id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={state.name} />

            <Flex direction='row' gap='size-200'>
                <TextField
                    flex='1'
                    label='Images folder path'
                    name='images_folder_path'
                    defaultValue={state.images_folder_path}
                />

                <Flex
                    height={'size-400'}
                    alignSelf={'end'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    UNSAFE_className={classes.folderIcon}
                >
                    <Folder />
                </Flex>
            </Flex>

            <Switch
                aria-label='ignore existing images'
                name='ignore_existing_images'
                defaultSelected={state.ignore_existing_images}
                key={state.ignore_existing_images ? 'true' : 'false'}
            >
                Ignore existing images
            </Switch>
        </Flex>
    );
};
