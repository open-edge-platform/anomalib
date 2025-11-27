import { usePipeline, useProjectIdentifier, useRunPipeline, useStopPipeline } from '@geti-inspect/hooks';
import { Flex, Switch } from '@geti/ui';
import isEmpty from 'lodash-es/isEmpty';
import { useWebRTCConnection } from 'src/components/stream/web-rtc-connection-provider';

import { isStatusActive } from '../../utils';

import classes from './pipeline-switch.module.scss';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const stopPipeline = useStopPipeline(projectId);
    const { status, start } = useWebRTCConnection();
    const { data: pipeline, isLoading } = usePipeline();

    const runPipeline = useRunPipeline({
        onSuccess: async () => {
            await start();
        },
    });

    const isSinkMissing = isEmpty(pipeline.sink?.id);
    const isModelMissing = isEmpty(pipeline.model?.id);
    const isPipelineActive = isStatusActive(pipeline.status);
    const isWebRtcConnecting = status === 'connecting';
    const isProcessing = runPipeline.isPending || stopPipeline.isPending;

    const handleChange = (isSelected: boolean) => {
        const handler = isSelected ? runPipeline.mutate : stopPipeline.mutate;

        handler({ params: { path: { project_id: projectId } } });
    };

    return (
        <Flex>
            <Switch
                UNSAFE_className={classes.switch}
                onChange={handleChange}
                isSelected={pipeline.status === 'running'}
                isDisabled={
                    isLoading ||
                    isProcessing ||
                    isSinkMissing ||
                    isModelMissing ||
                    !isPipelineActive ||
                    isWebRtcConnecting
                }
            >
                Enabled
            </Switch>
        </Flex>
    );
};
