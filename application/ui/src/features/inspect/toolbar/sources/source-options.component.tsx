import { ReactNode } from 'react';

import { useProjectIdentifier } from '@anomalib-studio/hooks';

import { ReactComponent as IpCameraIcon } from '../../../../assets/icons/ip-camera.svg';
import { ReactComponent as UsbCameraIcon } from '../../../../assets/icons/usb-camera.svg';
import { ReactComponent as Video } from '../../../../assets/icons/video-file.svg';
import { DisclosureGroup } from '../../../../components/disclosure-group/disclosure-group.component';
import { AddSource } from './add-source/add-source.component';
import { IpCameraFields } from './ip-camera/ip-camera-fields.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from './ip-camera/utils';
import { UsbCameraFields } from './usb-camera/usb-camera-fields.component';
import { getUsbCameraInitialConfig, usbCameraBodyFormatter } from './usb-camera/utils';
import { IPCameraSourceConfig, UsbCameraSourceConfig, VideoFileSourceConfig } from './util';
import { getVideoFileInitialConfig, videoFileBodyFormatter } from './video-file/utils';
import { VideoFileFields } from './video-file/video-file-fields.component';

interface SourceOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
}

export const SourceOptions = ({ hasHeader, children, onSaved }: SourceOptionsProps) => {
    const { projectId } = useProjectIdentifier();

    return (
        <>
            {hasHeader && children}

            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: 'USB Camera',
                        value: 'usb_camera',
                        icon: <UsbCameraIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getUsbCameraInitialConfig(projectId)}
                                componentFields={(state: UsbCameraSourceConfig) => (
                                    <UsbCameraFields defaultState={state} />
                                )}
                                bodyFormatter={usbCameraBodyFormatter}
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
                                config={getIpCameraInitialConfig(projectId)}
                                componentFields={(state: IPCameraSourceConfig) => (
                                    <IpCameraFields defaultState={state} />
                                )}
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
                                config={getVideoFileInitialConfig(projectId)}
                                componentFields={(state: VideoFileSourceConfig) => (
                                    <VideoFileFields defaultState={state} />
                                )}
                                bodyFormatter={videoFileBodyFormatter}
                            />
                        ),
                    },
                    // Temporary disable until Backed implementation is ready
                    /* {
                        label: 'Images folder',
                        value: 'images_folder',
                        icon: <ImageIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getImageFolderInitialConfig(projectId)}
                                componentFields={(state: ImagesFolderSourceConfig) => (
                                    <ImageFolderFields defaultState={state} />
                                )}
                                bodyFormatter={imageFolderBodyFormatter}
                            />
                        ),
                    }, */
                ]}
            />
        </>
    );
};
