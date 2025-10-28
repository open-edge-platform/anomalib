import { DisclosureGroup } from 'src/components/disclosure-group/disclosure-group.component';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { ReactComponent as GenICam } from '../../../../assets/icons/genicam.svg';
import { ReactComponent as Image } from '../../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../../assets/icons/webcam.svg';
import { ImageFolder } from './image-folder/image-folder.component';
import { IpCamera } from './ip-camera/ip-camera.component';
import { getImageFolderData, getIpCameraData, getVideoFileData, getWebcamData, SourceConfig } from './util';
import { VideoFile } from './video-file/video-file.component';
import { Webcam } from './webcam/webcam.component';

type SourceOptionsProps = {
    sources: SourceConfig[];
};

export const SourceOptions = ({ sources }: SourceOptionsProps) => {
    const pipeline = usePipeline();

    return (
        <DisclosureGroup
            defaultActiveInput={pipeline.data?.source?.source_type ?? null}
            items={[
                {
                    label: 'Webcam',
                    value: 'webcam',
                    icon: <WebcamIcon width={'24px'} />,
                    content: <Webcam config={getWebcamData(sources)} />,
                },
                {
                    label: 'IP Camera',
                    value: 'ip_camera',
                    icon: <IpCameraIcon width={'24px'} />,
                    content: <IpCamera config={getIpCameraData(sources)} />,
                },
                { label: 'GenICam', value: 'gen_i_cam', content: <ImageFolder />, icon: <GenICam width={'24px'} /> },
                {
                    label: 'Video file',
                    value: 'video_file',
                    content: <VideoFile config={getVideoFileData(sources)} />,
                    icon: <Video width={'24px'} />,
                },
                {
                    label: 'Images folder',
                    value: 'images_folder',
                    icon: <Image width={'24px'} />,
                    content: <ImageFolder config={getImageFolderData(sources)} />,
                },
            ]}
        />
    );
};
