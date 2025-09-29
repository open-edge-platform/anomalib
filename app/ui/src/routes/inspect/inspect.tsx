// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@geti/ui';

import { Sidebar } from '../../features/inspect/sidebar.component';
import { StreamContainer } from '../../features/inspect/stream/stream-container';
import { Toolbar } from '../../features/inspect/toolbar';

export const Inspect = () => {
    return (
        <Grid
            areas={['toolbar sidebar', 'canvas sidebar']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                gridTemplateColumns: 'auto min-content',
                height: '100%',
                overflow: 'hidden',
                gap: '1px',
            }}
        >
            <Toolbar />
            <StreamContainer />
            <Sidebar />
        </Grid>
    );
};
