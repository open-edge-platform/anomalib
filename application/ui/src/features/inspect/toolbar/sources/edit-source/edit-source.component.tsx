import { Button, View } from 'packages/ui';

import { ImageFolder } from '../image-folder/image-folder.component';
import { IpCamera } from '../ip-camera/ip-camera.component';
import { getImageFolderData, getIpCameraData, getVideoFileData, getWebcamData, SourceConfig } from '../util';
import { VideoFile } from '../video-file/video-file.component';
import { Webcam } from '../webcam/webcam.component';

import classes from './edit-source.module.scss';

interface EditSourceProps {
    config: SourceConfig;
}

export const EditSource = ({ config }: EditSourceProps) => {
    if (config.source_type === 'webcam') {
        return (
            <View UNSAFE_className={classes.container}>
                <Webcam
                    config={getWebcamData([config])}
                    renderButtons={(isPending) => (
                        <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                            update
                        </Button>
                    )}
                />
            </View>
        );
    }

    if (config.source_type === 'ip_camera') {
        return (
            <View UNSAFE_className={classes.container}>
                <IpCamera
                    config={getIpCameraData([config])}
                    renderButtons={(isPending) => (
                        <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                            update
                        </Button>
                    )}
                />
            </View>
        );
    }

    if (config.source_type === 'video_file') {
        return (
            <View UNSAFE_className={classes.container}>
                <VideoFile
                    config={getVideoFileData([config])}
                    renderButtons={(isPending) => (
                        <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                            update
                        </Button>
                    )}
                />
            </View>
        );
    }

    return (
        <View UNSAFE_className={classes.container}>
            <ImageFolder
                config={getImageFolderData([config])}
                renderButtons={(isPending) => (
                    <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                        update
                    </Button>
                )}
            />
        </View>
    );
};
