import { getApiUrl } from 'src/api/client';

import { isTauri } from '../../utils';

/**
 * Downloads a file from a URL. In Tauri, shows a save dialog to let the user choose the location.
 * In browser, uses the traditional download approach.
 */
export const downloadFile = async (url: string, name?: string): Promise<void> => {
    const fullUrl = getApiUrl(url);

    if (isTauri()) {
        try {
            const { save } = await import('@tauri-apps/plugin-dialog');
            const { writeFile } = await import('@tauri-apps/plugin-fs');

            // Fetch the file first
            const response = await fetch(fullUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch file: ${response.statusText}`);
            }

            const blob = await response.blob();

            // Open save dialog with suggested filename
            const filePath = await save({
                defaultPath: name,
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
            downloadFileBrowser(fullUrl, name);
        }
    } else {
        downloadFileBrowser(fullUrl, name);
    }
};

/**
 * Browser-based file download using anchor element
 */
const downloadFileBrowser = (url: string, name?: string): void => {
    const link = document.createElement('a');

    if (name) {
        link.download = name;
    }

    link.href = url;
    link.hidden = true;
    link.click();

    setTimeout(() => link.remove(), 100);
};
