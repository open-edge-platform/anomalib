// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../../api/client';
import { EditSourceForm } from './edit-source-form.component';
import { SourcesList } from './source-list/source-list.component';
import { SourceOptions } from './source-options.component';
import { SourceConfig } from './util';

export const SourceActions = () => {
    const { projectId } = useProjectIdentifier();
    const sourcesQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/sources', {
        params: { path: { project_id: projectId } },
    });

    const sources = sourcesQuery.data ?? [];
    const [view, setView] = useState<'list' | 'options' | 'edit'>(isEmpty(sources) ? 'options' : 'list');
    const [currentSource, setCurrentSource] = useState<SourceConfig | null>(null);

    const handleAddSource = () => {
        setView('options');
    };

    const handleEditSource = (source: SourceConfig) => {
        setView('edit');
        setCurrentSource(source);
    };

    if (view === 'edit' && !isEmpty(currentSource)) {
        return <EditSourceForm config={currentSource} onSaved={() => setView('list')} />;
    }

    if (view === 'list') {
        return <SourcesList sources={sources} onAddSource={handleAddSource} onEditSource={handleEditSource} />;
    }

    return <SourceOptions onSaved={() => setView('list')} />;
};
