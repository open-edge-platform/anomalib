// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti/ui';

import { ProgressBarItem } from './items/progressbar.component';

export const StatusBar = () => {
    return (
        <View gridArea={'statusbar'} backgroundColor={'gray-100'} width={'100%'} height={'30px'} overflow={'hidden'}>
            <Flex direction={'row'} gap={'size-100'} width={'100%'} justifyContent={'end'}>
                <ProgressBarItem />
            </Flex>
        </View>
    );
};
