import { UsbCameraSourceConfig } from '../util';

export const getUsbCameraInitialConfig = (projectId: string): UsbCameraSourceConfig => ({
    id: '',
    name: '',
    device_id: 0,
    project_id: projectId,
    source_type: 'usb_camera',
});

export const usbCameraBodyFormatter = (formData: FormData): UsbCameraSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'usb_camera',
    project_id: String(formData.get('project_id')),
    device_id: Number(formData.get('device_id')),
});
