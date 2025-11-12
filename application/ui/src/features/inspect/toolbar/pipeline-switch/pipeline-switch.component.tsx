import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Switch, toast } from '@geti/ui';
import { useWebRTCConnection } from 'src/components/stream/web-rtc-connection-provider';

import { useSelectedMediaItem } from '../../selected-media-item-provider.component';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const { status, start, stop } = useWebRTCConnection();
    const { onSetSelectedMediaItem } = useSelectedMediaItem();
    const { data: pipeline, isLoading } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const isWebRtcConnecting = status === 'connecting';

    const enablePipeline = $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable', {
        onSuccess: async () => {
            await start();
            onSetSelectedMediaItem(undefined);
        },
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
    const disablePipeline = $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable', {
        onSuccess: () => stop(),
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleChange = (isSelected: boolean) => {
        const handler = isSelected ? enablePipeline.mutate : disablePipeline.mutate;
        handler({ params: { path: { project_id: projectId } } });
    };

    return (
        <Switch
            onChange={handleChange}
            isSelected={pipeline.status === 'running'}
            isDisabled={isLoading || isWebRtcConnecting}
        >
            {isWebRtcConnecting ? 'Connecting...' : 'Enabled'}
        </Switch>
    );
};
