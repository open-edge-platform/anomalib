// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@geti/ui';

import { ImageInference } from '../../features/inspect/image-inference.component';
import { InferenceProvider } from '../../features/inspect/inference-provider.component';
import { SelectedMediaItemProvider } from '../../features/inspect/selected-media-item-provider.component';
import { Sidebar } from '../../features/inspect/sidebar.component';
/*import { StreamContainer } from '../../features/inspect/stream/stream-container';*/
import { Toolbar } from '../../features/inspect/toolbar';

export const Inspect = () => {
    return (
        <Grid
            areas={['toolbar sidebar', 'canvas sidebar']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) minmax(0, 1fr)',
                gridTemplateColumns: 'auto min-content',
                height: '100%',
                overflow: 'hidden',
                gap: '1px',
            }}
        >
            <InferenceProvider>
                <SelectedMediaItemProvider>
                    <Toolbar />
                    <ImageInference />
                    {/*<StreamContainer />*/}
                    <Sidebar />
                </SelectedMediaItemProvider>
            </InferenceProvider>
        </Grid>
    );
};
