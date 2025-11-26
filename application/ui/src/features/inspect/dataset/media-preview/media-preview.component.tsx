// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, dimensionValue, Divider, Grid, Header, Heading, View } from '@geti/ui';

import { MediaItem } from '../types';
import { InferenceOpacity } from './inference-opacity/inference-opacity.component';
import { InferenceResult } from './inference-result/inference-result.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';

type MediaPreviewProps = {
    selectedMediaItem: MediaItem;
    mediaItems: MediaItem[];
    onClose: () => void;
    onSetSelectedMediaItem: (mediaItem: MediaItem | undefined) => void;
};

export const MediaPreview = ({ selectedMediaItem, mediaItems, onClose, onSetSelectedMediaItem }: MediaPreviewProps) => {
    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>
            <Header>
                <InferenceOpacity />
            </Header>

            <Divider />

            <Content UNSAFE_style={{ backgroundColor: 'var(--spectrum-global-color-gray-50)' }}>
                <Grid
                    gap='size-125'
                    rows='auto'
                    width='100%'
                    height='100%'
                    columns='1fr 140px'
                    UNSAFE_style={{ padding: dimensionValue('size-125') }}
                    areas={['canvas sidebar', 'canvas sidebar']}
                >
                    <View gridArea={'canvas'} overflow={'hidden'}>
                        <InferenceResult selectedMediaItem={selectedMediaItem} />
                    </View>

                    <View gridArea={'sidebar'}>
                        <SidebarItems
                            mediaItems={mediaItems}
                            selectedMediaItem={selectedMediaItem}
                            onSetSelectedMediaItem={onSetSelectedMediaItem}
                        />
                    </View>
                </Grid>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
