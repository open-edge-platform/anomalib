// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Folder } from '@geti-inspect/icons';
import { Flex, TextField } from '@geti/ui';

import { VideoFileSourceConfig } from '../util';

import classes from './video-file-fields.module.scss';

type VideoFileFieldsProps = {
    defaultState: VideoFileSourceConfig;
};

export const VideoFileFields = ({ defaultState }: VideoFileFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField width='100%' label='Name' name='name' defaultValue={defaultState.name} />

            <Flex direction='row' gap='size-200'>
                <TextField
                    width='100%'
                    name='video_path'
                    label='Video file path'
                    defaultValue={String(defaultState.video_path)}
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
        </Flex>
    );
};
