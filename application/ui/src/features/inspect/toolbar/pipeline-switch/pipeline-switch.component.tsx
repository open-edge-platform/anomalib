import { useActivatePipeline, usePipeline, useProjectIdentifier, useRunPipeline } from '@geti-inspect/hooks';
import { Flex, Switch } from '@geti/ui';
import isEmpty from 'lodash-es/isEmpty';
import { useWebRTCConnection } from 'src/components/stream/web-rtc-connection-provider';

import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { isStatusActive } from '../../utils';
import { WebRTCConnectionStatus } from './web-rtc-connection-status.component';

import classes from './pipeline-switch.module.scss';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const { status, start } = useWebRTCConnection();
    const { onSetSelectedMediaItem } = useSelectedMediaItem();
    const { data: pipeline, isLoading } = usePipeline();

    const runPipeline = useRunPipeline({
        onSuccess: async () => {
            await start();
            onSetSelectedMediaItem(undefined);
        },
    });

    const hasSink = !isEmpty(pipeline?.sink);
    const hasSource = !isEmpty(pipeline?.source);
    const activatePipeline = useActivatePipeline({});
    const isPipelineActive = isStatusActive(pipeline.status);
    const isWebRtcConnecting = status === 'connecting';
    const isInferenceAvailable = !isEmpty(pipeline.model?.id);

    const handleChange = async (isSelected: boolean) => {
        const handler = isSelected ? runPipeline.mutateAsync : activatePipeline.mutateAsync;
        await handler({ params: { path: { project_id: projectId } } });
    };

    return (
        <Flex>
            <Switch
                UNSAFE_className={classes.switch}
                onChange={handleChange}
                isSelected={pipeline.status === 'running'}
                isDisabled={
                    !hasSink ||
                    isLoading ||
                    !hasSource ||
                    !isPipelineActive ||
                    isWebRtcConnecting ||
                    !isInferenceAvailable
                }
            >
                Enabled
            </Switch>
            <WebRTCConnectionStatus />
        </Flex>
    );
};
