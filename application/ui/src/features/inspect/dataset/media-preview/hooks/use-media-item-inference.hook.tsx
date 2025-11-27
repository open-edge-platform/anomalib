import { SchemaPredictionResponse } from '@geti-inspect/api/spec';
import { usePipeline } from '@geti-inspect/hooks';
import { skipToken, useQuery } from '@tanstack/react-query';

import { isNonEmptyString } from '../../../utils';
import { MediaItem } from '../../types';

const downloadImageAsFile = async (media: MediaItem) => {
    const response = await fetch(`/api/projects/${media.project_id}/images/${media.id}/full`);

    const blob = await response.blob();

    return new File([blob], media.filename, { type: blob.type });
};

export const useMediaItemInference = (selectedMediaItem: MediaItem) => {
    const { data: pipeline } = usePipeline();
    const selectedModelId = pipeline?.model?.id;

    return useQuery({
        queryKey: ['inference', selectedMediaItem.id, selectedModelId],
        queryFn: isNonEmptyString(selectedModelId)
            ? async (): Promise<SchemaPredictionResponse> => {
                  const file = await downloadImageAsFile(selectedMediaItem);

                  const formData = new FormData();
                  formData.append('file', file);

                  if (pipeline?.inference_device) {
                      formData.append('device', pipeline.inference_device);
                  }

                  const response = await fetch(
                      `/api/projects/${selectedMediaItem.project_id}/models/${selectedModelId}:predict`,
                      { method: 'POST', body: formData }
                  );

                  return response.json();
              }
            : skipToken,
    });
};
