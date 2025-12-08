import { Selection, View } from '@geti/ui';
import { GridLayoutOptions } from 'react-aria-components';
import { getThumbnailUrl } from 'src/features/inspect/utils';

import { GridMediaItem } from '../../../../..//components/virtualizer-grid-layout/grid-media-item/grid-media-item.component';
import { MediaThumbnail } from '../../../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { DeleteMediaItem } from '../../delete-dataset-item/delete-dataset-item.component';
import { MediaItem } from '../../types';

interface SidebarItemsProps {
    mediaItems: MediaItem[];
    selectedMediaItem: MediaItem;
    layoutOptions: GridLayoutOptions;
    onSelectedMediaItem: (mediaItem: string | null) => void;
}

export const SidebarItems = ({
    mediaItems,
    layoutOptions,
    selectedMediaItem,
    onSelectedMediaItem,
}: SidebarItemsProps) => {
    const selectedIndex = mediaItems.findIndex((item) => item.id === selectedMediaItem.id);

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;
        const mediaItem = mediaItems.find((item) => item.id === firstKey);

        onSelectedMediaItem(mediaItem?.id ?? null);
    };

    const handleDeletedItem = (deletedIds: string[]) => {
        if (deletedIds.includes(String(selectedMediaItem.id))) {
            const nextIndex = selectedIndex + 1;
            const isLastItemDeleted = nextIndex >= mediaItems.length;
            const newSelectedIndex = isLastItemDeleted ? selectedIndex - 1 : nextIndex;
            const newSelectedItem = mediaItems[newSelectedIndex];

            onSelectedMediaItem(newSelectedItem?.id ?? null);
        }
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
                        topRightElement={() => (
                            <DeleteMediaItem itemsIds={[String(item.id)]} onDeleted={handleDeletedItem} />
                        )}
                    />
                )}
            />
        </View>
    );
};
