import { getApiUrl } from '@anomalib-studio/api';

import { MediaItem } from '../../types';

export const downloadImageAsFile = async (media: MediaItem) => {
    const response = await fetch(getApiUrl(`/api/projects/${media.project_id}/images/${media.id}/full`));

    const blob = await response.blob();

    return new File([blob], media.filename, { type: blob.type });
};
