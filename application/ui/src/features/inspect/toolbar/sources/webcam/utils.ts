import { WebcamSourceConfig } from '../util';

export const webcamInitialConfig: WebcamSourceConfig = {
    id: '',
    name: '',
    source_type: 'webcam',
    device_id: 0,
};

export const webcamBodyFormatter = (formData: FormData): WebcamSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'webcam',
    device_id: Number(formData.get('device_id')),
});
