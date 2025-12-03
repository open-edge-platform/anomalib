// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { PipelineLink } from '@geti-inspect/icons';
import {
    Button,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Item,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';

import { SinkActions } from './sinks/sink-actions.component';
import { SourceActions } from './sources/source-actions.component';

const paddingStyle = {
    '--spectrum-dialog-padding-x': dimensionValue('size-300'),
    '--spectrum-dialog-padding-y': dimensionValue('size-300'),
} as CSSProperties;

export const InputOutputSetup = () => {
    return (
        <DialogTrigger type='popover'>
            <Button variant={'secondary'} UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                <PipelineLink fill='white' />
                <Text width={'auto'}>Pipeline configuration</Text>
            </Button>
            <Dialog minWidth={'size-6000'} UNSAFE_style={paddingStyle}>
                <Content>
                    <Tabs aria-label='Dataset import tabs' height={'100%'}>
                        <TabList>
                            <Item key='sources' textValue='Sources'>
                                <Text>Input</Text>
                            </Item>
                            <Item key='sinks' textValue='Sinks'>
                                <Text>Output</Text>
                            </Item>
                        </TabList>
                        <TabPanels>
                            <Item key='sources'>
                                <View marginTop={'size-200'}>
                                    <SourceActions />
                                </View>
                            </Item>
                            <Item key='sinks'>
                                <View marginTop={'size-200'}>
                                    <SinkActions />
                                </View>
                            </Item>
                        </TabPanels>
                    </Tabs>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
