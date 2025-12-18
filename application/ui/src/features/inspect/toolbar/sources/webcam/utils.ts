import { WebcamSourceConfig } from '../util';

export const getWebcamInitialConfig = (projectId: string): WebcamSourceConfig => ({
    id: '',
    name: '',
    device_id: 0,
    project_id: projectId,
    source_type: 'webcam',
});

export const webcamBodyFormatter = (formData: FormData): WebcamSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'webcam',
    project_id: String(formData.get('project_id')),
    device_id: Number(formData.get('device_id')),
});
