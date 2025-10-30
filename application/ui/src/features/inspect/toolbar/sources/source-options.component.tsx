import { DisclosureGroup } from 'src/components/disclosure-group/disclosure-group.component';

import { ReactComponent as ImageIcon } from '../../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../../assets/icons/webcam.svg';
import { AddSource } from './add-source/add-source.component';
import { ImageFolderFields } from './image-folder/image-folder-fields.component';
import { imageFolderBodyFormatter, imageFolderInitialConfig } from './image-folder/utils';
import { IpCameraFields } from './ip-camera/ip-camera-fields.component';
import { ipCameraBodyFormatter, ipCameraInitialConfig } from './ip-camera/utils';
import { ImagesFolderSourceConfig, IPCameraSourceConfig, VideoFileSourceConfig, WebcamSourceConfig } from './util';
import { videoFileBodyFormatter, videoFileInitialConfig } from './video-file/utils';
import { VideoFileFields } from './video-file/video-file-fields.component';
import { webcamBodyFormatter, webcamInitialConfig } from './webcam/utils';
import { WebcamFields } from './webcam/webcam-fields.component';

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
                    content: (
                        <AddSource
                            onSaved={onSaved}
                            config={webcamInitialConfig}
                            componentFields={(state: WebcamSourceConfig) => <WebcamFields state={state} />}
                            bodyFormatter={webcamBodyFormatter}
                        />
                    ),
                },
                {
                    label: 'IP Camera',
                    value: 'ip_camera',
                    icon: <IpCameraIcon width={'24px'} />,
                    content: (
                        <AddSource
                            onSaved={onSaved}
                            config={ipCameraInitialConfig}
                            componentFields={(state: IPCameraSourceConfig) => <IpCameraFields state={state} />}
                            bodyFormatter={ipCameraBodyFormatter}
                        />
                    ),
                },
                {
                    label: 'Video file',
                    value: 'video_file',
                    icon: <Video width={'24px'} />,
                    content: (
                        <AddSource
                            onSaved={onSaved}
                            config={videoFileInitialConfig}
                            componentFields={(state: VideoFileSourceConfig) => <VideoFileFields state={state} />}
                            bodyFormatter={videoFileBodyFormatter}
                        />
                    ),
                },
                {
                    label: 'Images folder',
                    value: 'images_folder',
                    icon: <ImageIcon width={'24px'} />,
                    content: (
                        <AddSource
                            onSaved={onSaved}
                            config={imageFolderInitialConfig}
                            componentFields={(state: ImagesFolderSourceConfig) => <ImageFolderFields state={state} />}
                            bodyFormatter={imageFolderBodyFormatter}
                        />
                    ),
                },
            ]}
        />
    );
};
