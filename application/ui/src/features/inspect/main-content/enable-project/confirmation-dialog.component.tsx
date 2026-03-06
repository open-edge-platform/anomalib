import { $api } from '@anomalib-studio/api';
import { useActivatePipeline, useDisablePipeline } from '@anomalib-studio/hooks';
import { AlertDialog } from '@geti/ui';

interface ConfirmationDialogProps {
    activeProjectId: string;
    currentProjectId: string;
}

export const ConfirmationDialog = ({ activeProjectId, currentProjectId }: ConfirmationDialogProps) => {
    const activePipeline = useActivatePipeline({});
    const disablePipeline = useDisablePipeline(activeProjectId);

    const activeProject = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: activeProjectId } },
    });

    const currentProject = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: currentProjectId } },
    });

    const isUpdating = activeProject.isLoading || currentProject.isLoading;

    const handleEnableProject = async () => {
        await disablePipeline.mutateAsync({
            params: { path: { project_id: activeProjectId } },
        });

        await activePipeline.mutateAsync({
            params: { path: { project_id: currentProjectId } },
        });
    };

    return (
        <AlertDialog
            variant={'confirmation'}
            cancelLabel={'Cancel'}
            onPrimaryAction={handleEnableProject}
            primaryActionLabel={isUpdating ? 'Activating...' : 'Activate project'}
            isPrimaryActionDisabled={isUpdating}
            title={`Activate project "${currentProject.data.name}"`}
        >
            By activating this project, the current active project &quot;{activeProject.data.name}&quot; and its
            respective inference workflow will be disabled. Changing your active project can influence any downstream
            processes that might be connected to the output generated from the current active project.
        </AlertDialog>
    );
};
