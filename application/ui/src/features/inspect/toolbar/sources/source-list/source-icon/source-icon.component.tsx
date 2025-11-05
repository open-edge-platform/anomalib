import { ImagesFolder, IpCamera, VideoFile, Webcam } from '@geti-inspect/icons';

interface SourceIconProps {
    type: string;
}

export const SourceIcon = ({ type }: SourceIconProps) => {
    if (type === 'webcam') {
        return <Webcam />;
    }

    if (type === 'ip_camera') {
        return <IpCamera />;
    }

    if (type === 'video_file') {
        return <VideoFile />;
    }

    return <ImagesFolder />;
};
