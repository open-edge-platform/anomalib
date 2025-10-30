// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { isEmpty } from 'lodash-es';

import { $api } from '../../../../api/client';
import { EditSourceFactory } from './edit-source-factory.component';
import { SourcesList } from './source-list.component';
import { SourceOptions } from './source-options.component';
import { SourceConfig } from './util';

export const SourceFactory = () => {
    const sourcesQuery = $api.useSuspenseQuery('get', '/api/sources');
    const sources = sourcesQuery.data ?? [];
    const [view, setView] = useState<'list' | 'options' | 'edit'>(isEmpty(sources) ? 'options' : 'list');
    const [currentSource, setCurrentSource] = useState<SourceConfig | null>(null);

    const handleAddSource = () => {
        setView('options');
    };

    const handleEditSourceFactory = (source: SourceConfig) => {
        setView('edit');
        setCurrentSource(source);
    };

    if (view === 'edit' && !isEmpty(currentSource)) {
        return <EditSourceFactory config={currentSource} onSaved={() => setView('list')} />;
    }

    if (view === 'list') {
        return (
            <SourcesList
                sources={sources}
                onAddSource={handleAddSource}
                onEditSourceFactory={handleEditSourceFactory}
            />
        );
    }

    return <SourceOptions onSaved={() => setView('list')} />;
};
