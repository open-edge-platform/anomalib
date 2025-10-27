// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import {
    Button,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Item,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';

import { ReactComponent as Camera } from '../../../assets/icons/pipeline-link.svg';

export const InputOutputSetup = () => {
    return (
        <DialogTrigger type='popover'>
            <Button width={'size-3000'} variant={'secondary'} UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                <Camera fill='white' />
                <Text width={'auto'}>Pipeline configuration</Text>
            </Button>
            <Dialog minWidth={'size-6000'}>
                <Content>
                    <Tabs aria-label='Dataset import tabs' height={'100%'}>
                        <TabList>
                            <Item key='sources' textValue='FoR'>
                                <Text>Input</Text>
                            </Item>
                            <Item key='sinks' textValue='MaR'>
                                <Text>Output</Text>
                            </Item>
                        </TabList>
                        <TabPanels>
                            <Item key='sources'>
                                <View marginTop={'size-200'}>
                                    <Suspense fallback={<Loading size='M' />}>SourceOptions</Suspense>
                                </View>
                            </Item>
                            <Item key='sinks'>
                                <View marginTop={'size-200'}>
                                    <Suspense fallback={<Loading size='M' />}>SinkOptions</Suspense>
                                </View>
                            </Item>
                        </TabPanels>
                    </Tabs>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
