import { EditSource } from './edit-source/edit-source.component';
import { ImageFolderFields } from './image-folder/image-folder-fields.component';
import { imageFolderBodyFormatter } from './image-folder/utils';
import { IpCameraFields } from './ip-camera/ip-camera-fields.component';
import { ipCameraBodyFormatter } from './ip-camera/utils';
import { ImagesFolderSourceConfig, SourceConfig, VideoFileSourceConfig } from './util';
import { videoFileBodyFormatter } from './video-file/utils';
import { VideoFileFields } from './video-file/video-file-fields.component';
import { webcamBodyFormatter } from './webcam/utils';
import { WebcamFields } from './webcam/webcam-fields.component';

interface EditSourceFormProps {
    config: SourceConfig;
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSourceForm = ({ config, onSaved, onBackToList }: EditSourceFormProps) => {
    if (config.source_type === 'webcam') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <WebcamFields defaultState={state} />}
                bodyFormatter={webcamBodyFormatter}
            />
        );
    }

    if (config.source_type === 'ip_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <IpCameraFields defaultState={state} />}
                bodyFormatter={ipCameraBodyFormatter}
            />
        );
    }

    if (config.source_type === 'video_file') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state: VideoFileSourceConfig) => <VideoFileFields defaultState={state} />}
                bodyFormatter={videoFileBodyFormatter}
            />
        );
    }

    return (
        <EditSource
            onSaved={onSaved}
            onBackToList={onBackToList}
            config={config as ImagesFolderSourceConfig}
            componentFields={(state: ImagesFolderSourceConfig) => <ImageFolderFields defaultState={state} />}
            bodyFormatter={imageFolderBodyFormatter}
        />
    );
};
