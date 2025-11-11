import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Switch, toast } from '@geti/ui';

export const PipelineSwitch = () => {
    const { projectId } = useProjectIdentifier();
    const { data: pipeline } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const enablePipeline = $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable', {
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
        <Switch isSelected={pipeline.status === 'running'} onChange={handleChange}>
            Enabled
        </Switch>
    );
};
