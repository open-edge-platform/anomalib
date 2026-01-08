import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

const REFETCH_INTERVAL_WITH_TRAINING = 5_000;

export const useProjectTrainingJobs = () => {
    const { projectId } = useProjectIdentifier();

    const { data } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: ({ state }) => {
            const projectHasTrainingJob = state.data?.jobs.some(
                ({ project_id, type, status }) =>
                    projectId === project_id && type === 'training' && (status === 'running' || status === 'pending')
            );

            return projectHasTrainingJob ? REFETCH_INTERVAL_WITH_TRAINING : undefined;
        },
    });

    return { jobs: data?.jobs.filter((job) => job.project_id === projectId) };
};
