// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton } from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';

import { MediaItem } from '../types';
import { downloadFile } from './utils';

import classes from './download-dataset-item.module.scss';

export interface DownloadDatasetItemProps {
    mediaItem: MediaItem;
}

export const DownloadDatasetItem = ({ mediaItem }: DownloadDatasetItemProps) => {
    const handleDownload = () => {
        const url = `/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/full`;
        downloadFile(url, mediaItem.filename);
    };

    return (
        <ActionButton
            isQuiet
            aria-label='download media item'
            UNSAFE_className={classes.downloadButton}
            onPress={handleDownload}
        >
            <DownloadIcon />
        </ActionButton>
    );
};
