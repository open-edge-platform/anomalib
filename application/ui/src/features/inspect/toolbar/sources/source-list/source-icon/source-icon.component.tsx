import { ImagesFolder, IpCamera, UsbCamera, VideoFile } from '@geti-inspect/icons';

interface SourceIconProps {
    type: string;
}

export const SourceIcon = ({ type }: SourceIconProps) => {
    if (type === 'usb_camera') {
        return <UsbCamera />;
    }

    if (type === 'ip_camera') {
        return <IpCamera />;
    }

    if (type === 'video_file') {
        return <VideoFile />;
    }

    return <ImagesFolder />;
};
