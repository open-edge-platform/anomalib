// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { $api } from '../../../../api/client';
import { SourcesList } from './source-list.compnent';
import { SourceOptions } from './source-options.component';

export const SourceFactory = () => {
    const sourcesQuery = $api.useSuspenseQuery('get', '/api/sources');
    const sources = sourcesQuery.data ?? [];

    return isEmpty(sources) ? <SourceOptions sources={sources} /> : <SourcesList sources={sources} />;
};
