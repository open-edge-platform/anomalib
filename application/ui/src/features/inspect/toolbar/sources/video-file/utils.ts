import { VideoFileSourceConfig } from '../util';

export const getVideoFileInitialConfig = (projectId: string): VideoFileSourceConfig => ({
    id: '',
    name: '',
    video_path: '',
    project_id: projectId,
    source_type: 'video_file',
});

export const videoFileBodyFormatter = (formData: FormData): VideoFileSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'video_file',
    project_id: String(formData.get('project_id')),
    video_path: String(formData.get('video_path')),
});
