import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, Switch, toast } from '@geti/ui';
import { useWebRTCConnection } from 'src/components/stream/web-rtc-connection-provider';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { WebRTCConnectionStatus } from './web-rtc-connection-status.component';

import classes from './pipeline-switch.module.scss';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const { status, start, stop } = useWebRTCConnection();
    const { onSetSelectedMediaItem } = useSelectedMediaItem();
    const { data: pipeline, isLoading } = usePipeline();

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

    const hasSink = pipeline?.sink !== undefined;
    const hasSource = pipeline?.source !== undefined;

    const handleChange = (isSelected: boolean) => {
        const handler = isSelected ? enablePipeline.mutate : disablePipeline.mutate;
        handler({ params: { path: { project_id: projectId } } });
    };

    return (
        <Flex>
            <Switch
                UNSAFE_className={classes.switch}
                onChange={handleChange}
                isSelected={pipeline.status === 'running'}
                isDisabled={isLoading || isWebRtcConnecting || !hasSink || !hasSource}
            >
                Enabled
            </Switch>
            <WebRTCConnectionStatus />
        </Flex>
    );
};
