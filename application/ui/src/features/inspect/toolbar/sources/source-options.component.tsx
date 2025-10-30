import { DisclosureGroup } from 'src/components/disclosure-group/disclosure-group.component';

import { ReactComponent as Image } from '../../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../../assets/icons/webcam.svg';
import { ImageFolder } from './image-folder/image-folder.component';
import { AddIpCamera } from './ip-camera/add-ip-camera.component';
import { VideoFile } from './video-file/video-file.component';
import { Webcam } from './webcam/webcam.component';

interface SourceOptionsProps {
    onSaved: () => void;
}

export const SourceOptions = ({ onSaved }: SourceOptionsProps) => {
    return (
        <DisclosureGroup
            defaultActiveInput={null}
            items={[
                {
                    label: 'Webcam',
                    value: 'webcam',
                    icon: <WebcamIcon width={'24px'} />,
                    content: <Webcam />,
                },
                {
                    label: 'IP Camera',
                    value: 'ip_camera',
                    icon: <IpCameraIcon width={'24px'} />,
                    content: <AddIpCamera onSaved={onSaved} />,
                },
                {
                    label: 'Video file',
                    value: 'video_file',
                    content: <VideoFile />,
                    icon: <Video width={'24px'} />,
                },
                {
                    label: 'Images folder',
                    value: 'images_folder',
                    icon: <Image width={'24px'} />,
                    content: <ImageFolder />,
                },
            ]}
        />
    );
};
