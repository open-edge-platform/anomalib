// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Button, Flex, Form, TextField } from '@geti/ui';
import { isEmpty, isFunction } from 'lodash-es';

import { ReactComponent as Folder } from '../../../../../assets/icons/folder.svg';
import { useSourceAction } from '../hooks/use-source-action.hook';
import { VideoFileSourceConfig } from '../util';

import classes from './video-file.module.scss';

type VideoFileProps = {
    config?: VideoFileSourceConfig;
    renderButtons?: (isPending: boolean) => ReactNode;
};

const initConfig: VideoFileSourceConfig = {
    id: '',
    name: '',
    source_type: 'video_file',
    video_path: '',
};

export const VideoFile = ({ config = initConfig, renderButtons }: VideoFileProps) => {
    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            source_type: 'video_file',
            video_path: String(formData.get('video_path')),
        }),
    });

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width='100%' label='Name' name='name' defaultValue={state?.name} />

                <Flex direction='row' gap='size-200'>
                    <TextField
                        width='100%'
                        name='video_path'
                        label='Video file path'
                        defaultValue={String(state?.video_path)}
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

                {isFunction(renderButtons) ? (
                    renderButtons(isPending)
                ) : (
                    <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                        Add & Connect
                    </Button>
                )}
            </Flex>
        </Form>
    );
};
