// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Grid } from '@geti/ui';

import { Footer } from '../../features/inspect/footer/footer.component';
import { InferenceProvider } from '../../features/inspect/inference-provider.component';
import { MediaActions } from '../../features/inspect/media-actions/media-actions.component';
import { SelectedMediaItemProvider } from '../../features/inspect/selected-media-item-provider.component';
import { Sidebar } from '../../features/inspect/sidebar.component';
import { Toolbar } from '../../features/inspect/toolbar/toolbar';

export const Inspect = () => {
    const { projectId } = useProjectIdentifier();

    return (
        <Grid
            areas={['toolbar sidebar', 'canvas sidebar', 'footer sidebar']}
            rows={['size-800', 'minmax(0, 1fr)', 'auto']}
            columns={['1fr', 'min-content']}
            height={'100%'}
            gap={'size-10'}
            UNSAFE_style={{
                overflow: 'hidden',
            }}
            key={projectId}
        >
            <SelectedMediaItemProvider>
                <InferenceProvider>
                    <Toolbar />
                    <MediaActions />
                    <Sidebar />
                    <Footer />
                </InferenceProvider>
            </SelectedMediaItemProvider>
        </Grid>
    );
};
