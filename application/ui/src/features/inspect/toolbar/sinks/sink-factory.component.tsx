// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { ActionButton, Flex, Text } from '@geti/ui';
import { Back } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { SinkOptions } from './sink-options.component';
import { SinkConfig } from './utils';

export const SinkFactory = () => {
    const sinksQuery = $api.useSuspenseQuery('get', '/api/sinks');
    const sources = sinksQuery.data ?? [];

    const [view, setView] = useState<'newItemsOptions' | 'list' | 'options' | 'edit'>(
        isEmpty(sources) ? 'options' : 'list'
    );
    const [currentSource, setCurrentSource] = useState<SinkConfig | null>(null);

    const handleShowList = () => {
        setView('list');
    };

    /* const handleAddSource = () => {
        setView('newItemsOptions');
    };

    const handleEditSourceFactory = (source: SinkConfig) => {
        setView('edit');
        setCurrentSource(source);
    }; */

    if (view === 'edit' && !isEmpty(currentSource)) {
        return <p>edith</p>;
    }

    if (view === 'list') {
        return <p>list</p>;
    }

    return (
        <SinkOptions onSaved={handleShowList} hasHeader={sources.length > 0}>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>Add new input source</Text>
            </Flex>
        </SinkOptions>
    );
};
