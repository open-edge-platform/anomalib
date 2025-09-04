// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Grid, Heading, View } from '@geti/ui';

import { ReactComponent as DoubleChevronRight } from './../../assets/icons/double-chevron-right-icon.svg';
import { SourceForm } from './source/source-form';

export const Aside = () => {
    const [isHidden, setIsHidden] = useState(false);

    return (
        <Grid
            gridArea={'aside'}
            height={'90vh'}
            areas={['header', 'graphs']}
            rows={['min-content', 'minmax(0, 1fr)']}
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-200)',
                paddingLeft: isHidden
                    ? 'var(--spectrum-global-dimension-size-100)'
                    : 'var(--spectrum-global-dimension-size-200)',
                paddingRight: isHidden
                    ? 'var(--spectrum-global-dimension-size-100)'
                    : 'var(--spectrum-global-dimension-size-200)',
                backgroundColor: 'var(--spectrum-global-color-gray-100)',
                transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                width: isHidden
                    ? 'var(--spectrum-global-dimension-size-400)'
                    : 'var(--spectrum-global-dimension-size-6000)',
            }}
        >
            <Flex gridArea={'header'} alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                <ActionButton
                    isQuiet
                    onPress={() => setIsHidden((hidden) => !hidden)}
                    UNSAFE_style={{
                        transform: isHidden ? 'scaleX(-1)' : 'scaleX(1)',
                        cursor: 'pointer',
                    }}
                >
                    <DoubleChevronRight />
                </ActionButton>
                <Heading level={4} isHidden={isHidden}>
                    Input source
                </Heading>
            </Flex>
            <View gridArea={'graphs'} isHidden={isHidden} UNSAFE_style={{ overflow: 'hidden auto' }}>
                <SourceForm />
            </View>
        </Grid>
    );
};
