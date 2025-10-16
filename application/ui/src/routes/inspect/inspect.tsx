// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Grid } from '@geti/ui';

import { InferenceProvider } from '../../features/inspect/inference-provider.component';
import { InferenceResult } from '../../features/inspect/inference-result.component';
import { SelectedMediaItemProvider } from '../../features/inspect/selected-media-item-provider.component';
import { Sidebar } from '../../features/inspect/sidebar.component';
import { Toolbar } from '../../features/inspect/toolbar';

export const Inspect = () => {
    const { projectId } = useProjectIdentifier();

    return (
        <Grid
            areas={['toolbar sidebar', 'canvas sidebar']}
            rows={['size-800', 'minmax(0, 1fr)']}
            columns={['1fr', 'min-content']}
            height={'100%'}
            gap={'size-10'}
            UNSAFE_style={{
                overflow: 'hidden',
            }}
            key={projectId}
        >
            <InferenceProvider>
                <SelectedMediaItemProvider>
                    <Toolbar />
                    <InferenceResult />
                    <Sidebar />
                </SelectedMediaItemProvider>
            </InferenceProvider>
        </Grid>
    );
};
