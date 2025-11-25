// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, dimensionValue, Divider, Flex, Grid, Heading, View } from '@geti/ui';

import { MediaItem } from '../types';
import { InferenceResult } from './inference-result/inference-result.component';

type MediaPreviewProps = {
    projectId: string;
    mediaItem: MediaItem;
    onClose: () => void;
    onSelectedMediaItem: (item: MediaItem) => void;
};

export const MediaPreview = ({ mediaItem, projectId, onClose, onSelectedMediaItem }: MediaPreviewProps) => {
    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>

            <Divider />

            <Content
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-50)',
                }}
            >
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    columns='1fr 140px'
                    UNSAFE_style={{ padding: dimensionValue('size-125') }}
                    areas={['canvas aside', 'canvas aside']}
                >
                    <View gridArea={'canvas'} overflow={'hidden'}>
                        <InferenceResult selectedMediaItem={mediaItem} />
                    </View>

                    {/* <View gridArea={'aside'}>
                            <SidebarItems
                                items={items}
                                mediaItem={mediaItem}
                                hasNextPage={hasNextPage}
                                isFetchingNextPage={isFetchingNextPage}
                                fetchNextPage={fetchNextPage}
                                onSelectedMediaItem={onSelectedMediaItem}
                            />
                        </View> */}
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
