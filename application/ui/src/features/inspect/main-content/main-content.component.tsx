import { usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import isEmpty from 'lodash-es/isEmpty';

import { StreamContainer } from '../stream/stream-container';
import { EnableProject } from './enable-project/enable-project.component';
import { useEnsureActivePipeline } from './hooks/use-ensure-active-pipeline.hook';
import { SourceSinkMessage } from './source-sink-message/source-sink-message.component';

export const MainContent = () => {
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const { hasActiveProject, isCurrentProjectActive, activeProjectId } = useEnsureActivePipeline(projectId);

    if (isEmpty(pipeline.source?.id)) {
        return <SourceSinkMessage />;
    }

    if (hasActiveProject && !isCurrentProjectActive) {
        return <EnableProject currentProjectId={projectId} activeProjectId={String(activeProjectId)} />;
    }

    return <StreamContainer />;
};
