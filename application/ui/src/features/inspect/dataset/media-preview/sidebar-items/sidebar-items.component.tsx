import { Selection, Size, View } from '@geti/ui';
import { getThumbnailUrl } from 'src/features/inspect/utils';

import { GridMediaItem } from '../../../../..//components/virtualizer-grid-layout/grid-media-item/grid-media-item.component';
import { MediaThumbnail } from '../../../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { MediaItem } from '../../types';

interface SidebarItemsProps {
    mediaItems: MediaItem[];
    selectedMediaItem: MediaItem;
    onSelectedMediaItem: (mediaItem: string | null) => void;
}

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(120, 120),
    maxItemSize: new Size(120, 120),
    preserveAspectRatio: true,
};

export const SidebarItems = ({ mediaItems, selectedMediaItem, onSelectedMediaItem }: SidebarItemsProps) => {
    const selectedIndex = mediaItems.findIndex((item) => item.id === selectedMediaItem.id);

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;
        const mediaItem = mediaItems.find((item) => item.id === firstKey);

        onSelectedMediaItem(mediaItem?.id ?? null);
    };

    return (
        <View width={'100%'} height={'100%'}>
            <VirtualizerGridLayout
                items={mediaItems}
                ariaLabel='sidebar-items'
                selectionMode='single'
                selectedKeys={new Set([String(selectedMediaItem.id)])}
                layoutOptions={layoutOptions}
                scrollToIndex={selectedIndex}
                onSelectionChange={handleSelectionChange}
                contentItem={(item) => (
                    <GridMediaItem
                        contentElement={() => (
                            <MediaThumbnail
                                alt={item.filename}
                                url={getThumbnailUrl(item)}
                                onClick={() => onSelectedMediaItem(item.id ?? null)}
                            />
                        )}
                    />
                )}
            />
        </View>
    );
};
