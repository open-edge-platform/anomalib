// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@geti/ui';

import { Aside } from '../../features/inference/aside';
import { StreamContainer } from '../../features/inference/stream/stream-container';
import { Toolbar } from '../../features/inference/toolbar';

export const Inference = () => {
    return (
        <Grid
            areas={['toolbar aside', 'canvas aside']}
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
            <Aside />
        </Grid>
    );
};
