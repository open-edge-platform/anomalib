import { isTauri } from '@tauri-apps/api/core';

import { getApiUrl } from 'src/api/client';

import { downloadBlob } from '../../utils';

/**
 * Downloads a file from a URL.
 *
 * - In browsers, prefer a direct `<a href>` download to allow streaming and
 *   avoid loading the entire file into memory.
 * - In Tauri (or when no DOM is available), fetch the file as a Blob and
 *   delegate to `downloadBlob` for platform-specific handling.
 */
export const downloadFile = async (url: string, name?: string): Promise<void> => {
    const fullUrl = getApiUrl(url);

    const hasDOM = typeof document !== 'undefined' && typeof document.createElement === 'function';

    // In a regular browser environment, use a direct anchor-based download so the
    // browser can stream the response and we don't duplicate the payload in memory.
    if (hasDOM && !isTauri()) {
        const link = document.createElement('a');
        link.href = fullUrl;

        if (name) {
            link.download = name;
        }

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        return;
    }

    const response = await fetch(fullUrl);
    if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.statusText}`);
    }

    const blob = await response.blob();

    const contentDisposition = response.headers.get('Content-Disposition') ?? '';
    let filenameFromHeader: string | undefined;
    // Prefer RFC 5987 filename* (e.g. filename*=UTF-8''foo%20bar.zip) and decode it.
    const filenameStarMatch = contentDisposition.match(/filename\*\s*=\s*(?:UTF-8'[^']*')?("?)([^";]+)\1/i);
    if (filenameStarMatch && filenameStarMatch[2]) {
        const encodedFilename = filenameStarMatch[2];
        try {
            filenameFromHeader = decodeURIComponent(encodedFilename);
        } catch {
            // Fallback to the raw encoded value if decoding fails.
            filenameFromHeader = encodedFilename;
        }
    } else {
        // Fallback to legacy filename= parameter.
        const filenameMatch = contentDisposition.match(/filename\s*=\s*"?([^\";]+)"?/i);
        filenameFromHeader = filenameMatch?.[1];
    }

    const filename = name ?? filenameFromHeader ?? fullUrl.split('/').pop() ?? 'download';

    await downloadBlob(blob, filename);
};
