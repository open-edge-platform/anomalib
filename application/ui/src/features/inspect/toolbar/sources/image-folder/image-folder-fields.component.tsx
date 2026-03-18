// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Folder } from '@anomalib-studio/icons';
import { ActionButton, Flex, Switch, TextField, toast } from '@geti/ui';

import { ImagesFolderSourceConfig } from '../util';

import classes from './image-folder-fields.module.scss';

type ImageFolderFieldsProps = {
    defaultState: ImagesFolderSourceConfig;
};

export const ImageFolderFields = ({ defaultState }: ImageFolderFieldsProps) => {
    const directoryInputRef = useRef<HTMLInputElement | null>(null);

    const showFolderHelp = () => {
        toast({
            title: 'Folder selection',
            type: 'info',
            message:
                "A folder picker can help you choose a local folder, but the backend can only read folders that exist on the server (or inside the Docker container). " +
                "If you're running in Docker, mount the folder into the container and enter the container path here (for example: /data/images).",
            duration: 8000,
            position: 'bottom-left',
            actionButtons: [],
        });
    };

    const openFolderPicker = () => {
        // Best-effort: open a local folder picker (supported in Chromium-based browsers).
        // It won't provide an absolute path the backend can read, but users expect a picker.
        const input = directoryInputRef.current;
        if (!input) {
            showFolderHelp();
            return;
        }
        try {
            input.click();
        } catch {
            showFolderHelp();
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={defaultState.name} />

            <Flex direction='row' gap='size-200'>
                <TextField
                    flex='1'
                    label='Images folder path'
                    name='images_folder_path'
                    defaultValue={defaultState.images_folder_path}
                />

                <input
                    ref={directoryInputRef}
                    // non-standard attributes supported by Chromium-based browsers
                    {...({ webkitdirectory: '', directory: '' } as any)}
                    type='file'
                    multiple
                    hidden
                    onChange={(e) => {
                        const count = e.currentTarget.files?.length ?? 0;
                        if (count === 0) {
                            showFolderHelp();
                            return;
                        }
                        toast({
                            title: 'Folder selected',
                            type: 'info',
                            message: `Selected ${count} file(s). Now enter the folder path that the backend can read (for Docker, typically a mounted path like /data/images).`,
                            duration: 8000,
                            position: 'bottom-left',
                            actionButtons: [],
                        });
                        // allow selecting the same folder again
                        e.currentTarget.value = '';
                    }}
                />

                <ActionButton
                    aria-label='Choose a folder'
                    alignSelf={'end'}
                    height={'size-400'}
                    UNSAFE_className={classes.folderIcon}
                    onPress={openFolderPicker}
                >
                    <Folder />
                </ActionButton>
            </Flex>

            <Switch
                aria-label='ignore existing images'
                name='ignore_existing_images'
                defaultSelected={defaultState.ignore_existing_images}
                key={defaultState.ignore_existing_images ? 'true' : 'false'}
            >
                Ignore existing images
            </Switch>
        </Flex>
    );
};