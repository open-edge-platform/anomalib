import { usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
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

    const showEnableProject = hasActiveProject && !isCurrentProjectActive;

    return (
        <>
            {showEnableProject && (
                <EnableProject currentProjectId={projectId} activeProjectId={String(activeProjectId)} />
            )}
            <div style={{ display: showEnableProject ? 'none' : 'contents' }}>
                <StreamContainer />
            </div>
        </>
    );
};
