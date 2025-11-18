import { useActivePipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import isEmpty from 'lodash-es/isEmpty';

import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { StreamContainer } from '../stream/stream-container';
import { EnableProject } from './enable-project/enable-project.component';
import { InferenceResult } from './inference-result/inference-result.component';
import { SourceSinkMessage } from './source-sink-message/source-sink-message.component';

export const MainContent = () => {
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const { selectedMediaItem } = useSelectedMediaItem();
    const { data: activeProjectPipeline } = useActivePipeline();

    const hasActiveProject = !isEmpty(activeProjectPipeline);
    const isMissingSourceOrSink = isEmpty(pipeline.sink?.id) || isEmpty(pipeline.source?.id);
    const isCurrentProjectActive = activeProjectPipeline?.project_id === projectId;

    if (isEmpty(selectedMediaItem) && isMissingSourceOrSink) {
        return <SourceSinkMessage />;
    }

    if (isEmpty(selectedMediaItem) && hasActiveProject && !isCurrentProjectActive) {
        return <EnableProject currentProjectId={projectId} activeProjectId={activeProjectPipeline.project_id} />;
    }

    if (isEmpty(selectedMediaItem)) {
        return <StreamContainer />;
    }

    return <InferenceResult selectedMediaItem={selectedMediaItem} />;
};
