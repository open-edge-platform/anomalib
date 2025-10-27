// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Form, Switch, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ReactComponent as Folder } from '../../../../../assets/icons/folder.svg';
import { useSourceAction } from '../hooks/use-source-action.hook';
import { ImagesFolderSourceConfig } from '../util';

import classes from './image-folder.module.scss';

type ImageFolderProps = {
    config?: ImagesFolderSourceConfig;
};
const iniConfig: ImagesFolderSourceConfig = {
    id: '',
    name: '',
    source_type: 'images_folder',
    images_folder_path: '',
    ignore_existing_images: false,
};

export const ImageFolder = ({ config = iniConfig }: ImageFolderProps) => {
    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            source_type: 'images_folder',
            images_folder_path: String(formData.get('images_folder_path')),
            ignore_existing_images: formData.get('ignore_existing_images') === 'on' ? true : false,
        }),
    });

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <Flex direction='row' gap='size-200'>
                    <TextField
                        flex='1'
                        label='Images folder path'
                        name='images_folder_path'
                        defaultValue={state?.images_folder_path}
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
                    defaultSelected={state?.ignore_existing_images}
                    key={state?.ignore_existing_images ? 'true' : 'false'}
                >
                    Ignore existing images
                </Switch>

                <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
