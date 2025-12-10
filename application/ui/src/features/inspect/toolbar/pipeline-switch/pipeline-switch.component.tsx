import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@geti-inspect/hooks';
import { Flex, Switch } from '@geti/ui';

import classes from './pipeline-switch.module.scss';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const { data: pipeline } = usePipeline();

    const hasOverlay = pipeline?.overlay ?? false;
    const isPipelineStopped = pipeline?.status !== 'running';

    const handleChange = () => {
        patchPipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { overlay: !hasOverlay } });
    };

    return (
        <Flex>
            <Switch
                UNSAFE_className={classes.switch}
                onChange={handleChange}
                isSelected={hasOverlay}
                isDisabled={patchPipeline.isPending || isPipelineStopped}
            >
                Overlay
            </Switch>
        </Flex>
    );
};
