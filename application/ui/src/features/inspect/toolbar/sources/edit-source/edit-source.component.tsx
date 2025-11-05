// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { Button, ButtonGroup, Divider, Form, View } from '@geti/ui';

import { useConnectSourceToPipeline } from '../../../../../hooks/use-pipeline.hook';
import { useSourceAction } from '../hooks/use-source-action.hook';
import { SourceConfig } from '../util';

import classes from './edit-source.module.scss';

interface EditSourceProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
}

export const EditSource = <T extends SourceConfig>({
    config,
    onSaved,
    bodyFormatter,
    componentFields,
}: EditSourceProps<T>) => {
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: false,
        onSaved: async (sourceId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sourceId));
            connectToPipeline.current = false;
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form action={submitAction}>
            <View UNSAFE_className={classes.container}>
                <>{componentFields(state)}</>
            </View>
            <Divider size='S' marginY={'size-200'} />

            <ButtonGroup marginTop={'0px'}>
                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = false)}
                >
                    Save
                </Button>

                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = true)}
                >
                    Save & Connect
                </Button>
            </ButtonGroup>
        </Form>
    );
};
