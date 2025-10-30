import { EditSource } from './edit-source/edit-source.component';
import { ImageFolderFields } from './image-folder/image-folder-fields.component';
import { imageFolderBodyFormatter } from './image-folder/utils';
import { IpCameraFields } from './ip-camera/ip-camera-fields.component';
import { ipCameraBodyFormatter } from './ip-camera/utils';
import {
    getImageFolderData,
    getIpCameraData,
    getVideoFileData,
    getWebcamData,
    ImagesFolderSourceConfig,
    SourceConfig,
    VideoFileSourceConfig,
} from './util';
import { videoFileBodyFormatter } from './video-file/utils';
import { VideoFileFields } from './video-file/video-file-fields.component';
import { webcamBodyFormatter } from './webcam/utils';
import { WebcamFields } from './webcam/webcam-fields.component';

interface EditSourceFactoryProps {
    config: SourceConfig;
    onSaved: () => void;
}

export const EditSourceFactory = ({ config, onSaved }: EditSourceFactoryProps) => {
    if (config.source_type === 'webcam') {
        return (
            <EditSource
                onSaved={onSaved}
                config={getWebcamData([config])}
                componentFields={(state) => <WebcamFields state={state} />}
                bodyFormatter={webcamBodyFormatter}
            />
        );
    }

    if (config.source_type === 'ip_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={getIpCameraData([config])}
                componentFields={(state) => <IpCameraFields state={state} />}
                bodyFormatter={ipCameraBodyFormatter}
            />
        );
    }

    if (config.source_type === 'video_file') {
        return (
            <EditSource
                onSaved={onSaved}
                config={getVideoFileData([config])}
                componentFields={(state: VideoFileSourceConfig) => <VideoFileFields state={state} />}
                bodyFormatter={videoFileBodyFormatter}
            />
        );
    }

    return (
        <EditSource
            onSaved={onSaved}
            config={getImageFolderData([config])}
            componentFields={(state: ImagesFolderSourceConfig) => <ImageFolderFields state={state} />}
            bodyFormatter={imageFolderBodyFormatter}
        />
    );
};
