import { createContext, ReactNode, use, useState } from 'react';

import { MediaItem } from './dataset/types';

interface SelectedMediaItemContextProps {
    selectedMediaItem: MediaItem | undefined;
    onSetSelectedMediaItem: (mediaItem: MediaItem | undefined) => void;
}

const SelectedMediaItemContext = createContext<SelectedMediaItemContextProps | undefined>(undefined);

interface SelectedMediaItemProviderProps {
    children: ReactNode;
}

export const SelectedMediaItemProvider = ({ children }: SelectedMediaItemProviderProps) => {
    const [selectedMediaItem, setSelectedMediaItem] = useState<MediaItem | undefined>(undefined);

    return (
        <SelectedMediaItemContext value={{ selectedMediaItem, onSetSelectedMediaItem: setSelectedMediaItem }}>
            {children}
        </SelectedMediaItemContext>
    );
};

export const useSelectedMediaItem = () => {
    const context = use(SelectedMediaItemContext);

    if (context === undefined) {
        throw new Error('useSelectedMediaItem must be used within a SelectedMediaItemProvider');
    }

    return context;
};
