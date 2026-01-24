import { $api } from '@geti-inspect/api';

import { SourceConfig } from '../../util';

import classes from './settings-list.module.scss';

interface SettingsListProps {
    source: SourceConfig;
}

const CameraDeviceDisplay = ({ deviceId }: { deviceId: number }) => {
    const { data: cameraDevices, isLoading } = $api.useQuery('get', '/api/system/devices/camera');
    const devices = cameraDevices ?? [];
    const device = devices.find(({ index }) => index === deviceId);

    if (isLoading) {
        return <span>Loading...</span>;
    }

    return (
        <ul className={classes.list}>
            <li>Device: {device ? device.name : `Unknown device (id: ${deviceId})`}</li>
        </ul>
    );
};

export const SettingsList = ({ source }: SettingsListProps) => {
    if (source.source_type === 'images_folder') {
        return (
            <ul className={classes.list}>
                <li>Folder path: {source.images_folder_path}</li>
                <li>Ignore existing images: {source.ignore_existing_images ? 'Yes' : 'No'}</li>
            </ul>
        );
    }

    if (source.source_type === 'ip_camera') {
        return (
            <ul className={classes.list}>
                <li>Stream url: {source.stream_url}</li>
                <li>Auth required: {source.auth_required ? 'Yes' : 'No'}</li>
            </ul>
        );
    }

    if (source.source_type === 'video_file') {
        return (
            <ul className={classes.list}>
                <li>Video path: {source.video_path}</li>
            </ul>
        );
    }

    if (source.source_type === 'usb_camera') {
        return <CameraDeviceDisplay deviceId={source.device_id} />;
    }

    return <></>;
};
