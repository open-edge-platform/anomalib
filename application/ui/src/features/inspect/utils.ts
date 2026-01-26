import { isString } from 'lodash-es';

import { getApiUrl } from 'src/api/client';

import { MediaItem } from './dataset/types';

export const removeUnderscore = (text: string) => {
    return text.replaceAll('_', ' ');
};

export const isStatusActive = (status: string) => {
    return ['running', 'active'].includes(status);
};

/**
 * Check if the app is running inside Tauri v2
 */
export const isTauri = (): boolean => {
    return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
};

/**
 * Downloads a blob as a file. In Tauri, shows a save dialog to let the user choose the location.
 * In browser, uses the traditional download approach.
 */
export const downloadBlob = async (blob: Blob, filename: string): Promise<void> => {
    if (isTauri()) {
        console.debug('Downloading via Tauri');
        try {
            const { save } = await import('@tauri-apps/plugin-dialog');
            const { writeFile } = await import('@tauri-apps/plugin-fs');

            // Open save dialog with suggested filename
            const filePath = await save({
                defaultPath: filename,
                filters: [
                    {
                        name: 'All Files',
                        extensions: ['*'],
                    },
                ],
            });

            if (filePath) {
                // Convert blob to Uint8Array and write to file
                const arrayBuffer = await blob.arrayBuffer();
                const contents = new Uint8Array(arrayBuffer);
                await writeFile(filePath, contents);
            }
        } catch (error) {
            console.error('Failed to save file via Tauri:', error);
            // Fallback to browser download if Tauri save fails
            downloadBlobBrowser(blob, filename);
        }
    } else {
        console.debug('Downloading via Browser');
        downloadBlobBrowser(blob, filename);
    }
};

/**
 * Browser-based blob download using anchor element
 */
const downloadBlobBrowser = (blob: Blob, filename: string): void => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
};

export const sanitizeFilename = (name: string): string => {
    return name
        .replace(/\s+/g, '_')
        .replace(/[^a-zA-Z0-9_\-\.]/g, '')
        .toLowerCase();
};

export const formatSize = (bytes: number | null | undefined) => {
    if (bytes === null || bytes === undefined) {
        return '';
    }

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    const maximumFractionDigits = size >= 10 ? 0 : 1;
    const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits });

    return `${formatter.format(size)} ${units[unitIndex]}`;
};

export const isNonEmptyString = (value: unknown): value is string => isString(value) && value !== '';

export const getThumbnailUrl = (mediaItem: MediaItem) =>
    getApiUrl(`/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail`);

export const formatDuration = (seconds: number | null): string | null => {
    if (seconds === null) return null;

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
};
