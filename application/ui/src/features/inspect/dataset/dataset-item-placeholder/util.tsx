import { MediaItem } from '../types';

const PLACEHOLDER_FILENAME = 'placeholder';

export const isPlaceholderItem = (name: string): boolean => {
    return name.includes(PLACEHOLDER_FILENAME);
};

export const getPlaceholderItem = (index: number): MediaItem => {
    return {
        id: `${PLACEHOLDER_FILENAME}-${index}`,
        filename: PLACEHOLDER_FILENAME,
        project_id: '',
        size: 0,
        is_anomalous: false,
        width: 0,
        height: 0,
    };
};

export const getPlaceholderKeys = (count: number): string[] => {
    return Array.from({ length: count }, (_, index) => `${PLACEHOLDER_FILENAME}-${index}`);
};
