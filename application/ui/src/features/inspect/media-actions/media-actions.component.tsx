import { $api } from '@geti-inspect/api';
import { useActivePipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import isEmpty from 'lodash-es/isEmpty';

import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { StreamContainer } from '../stream/stream-container';
import { EnableProject } from './enable-project/enable-project.component';
import { InferenceResult } from './inference-result/inference-result.component';
import { SourceSinkMessage } from './source-sink-message/source-sink-message.component';
import { TrainModelMessage } from './train-model-message/train-model-message.component';

const useIsInferenceAvailable = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });

    return data?.models.length !== 0;
};

export const MediaActions = () => {
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const { selectedMediaItem } = useSelectedMediaItem();
    const isInferenceAvailable = useIsInferenceAvailable();
    const { data: activeProjectPipeline } = useActivePipeline();

    const hasSink = !isEmpty(pipeline.sink?.id);
    const hasSource = !isEmpty(pipeline.source?.id);
    const hasActiveProject = !isEmpty(activeProjectPipeline);
    const isCurrentProjectActive = activeProjectPipeline?.project_id === projectId;

    if (isEmpty(selectedMediaItem) && (!hasSource || !hasSink)) {
        return <SourceSinkMessage />;
    }

    if (isEmpty(selectedMediaItem) && !isInferenceAvailable) {
        return <TrainModelMessage />;
    }

    if (isEmpty(selectedMediaItem) && hasActiveProject && !isCurrentProjectActive) {
        return <EnableProject currentProjectId={projectId} activeProjectId={activeProjectPipeline.project_id} />;
    }

    if (isEmpty(selectedMediaItem)) {
        return <StreamContainer />;
    }

    return <InferenceResult selectedMediaItem={selectedMediaItem} />;
};
