// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Divider, Flex, Loading, NumberField, Text, TextField } from '@geti/ui';

import {
    SchemaDisconnectedSourceConfig,
    SchemaImagesFolderSourceConfig,
    SchemaIpCameraSourceConfig,
    SchemaVideoFileSourceConfig,
    SchemaWebcamSourceConfig,
} from '../../../api/openapi-spec';
import { ReactComponent as CameraOff } from '../../../assets/icons/camera-off.svg';
import { ReactComponent as Image } from '../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCamera } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as Webcam } from '../../../assets/icons/webcam.svg';
import { RadioDisclosure } from '../../../components/radio-disclosure-group/radio-disclosure-group';
import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { Stream } from '../stream/stream';

import classes from '../inference.module.scss';

type SourceConfig =
    | SchemaDisconnectedSourceConfig
    | SchemaImagesFolderSourceConfig
    | SchemaIpCameraSourceConfig
    | SchemaVideoFileSourceConfig
    | SchemaWebcamSourceConfig;

type SourceType = SourceConfig['source_type'];

type ConfigByDestinationType<T extends SourceType> = Extract<SourceConfig, { source_type: T }>;
type SourceFormRecord = {
    [SourceTypeKey in SourceType]: ConfigByDestinationType<SourceTypeKey>;
};

const DEFAULT_SOURCE_FORMS: SourceFormRecord = {
    disconnected: {
        name: 'Disconnected',
        source_type: 'disconnected',
    },
    images_folder: {
        name: 'Images folder',
        source_type: 'images_folder',
        images_folder_path: '',
        ignore_existing_images: false,
    },
    ip_camera: {
        name: 'Ip camera',
        source_type: 'ip_camera',
        stream_url: '',
        auth_required: false,
    },
    video_file: {
        name: 'Video file',
        source_type: 'video_file',
        video_path: '',
    },
    webcam: {
        name: 'Webcam',
        source_type: 'webcam',
        device_id: 0,
    },
};

export const ConnectionPreview = () => {
    const [size, setSize] = useState({ height: 608, width: 892 });
    const { status } = useWebRTCConnection();

    return (
        <Flex
            alignItems={'center'}
            justifyContent={'center'}
            width={'100%'}
            height={'size-3000'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
            }}
        >
            {status === 'idle' && (
                <div className={classes.canvasContainer}>
                    <Flex direction={'column'} justifyContent={'center'} alignItems={'center'} gap={'size-200'}>
                        <CameraOff />
                        <Text
                            UNSAFE_style={{
                                color: 'var(--spectrum-global-color-gray-700)',
                                fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                                lineHeight: 'var(--spectrum-global-dimension-size-225)',
                            }}
                        >
                            Configure your input source
                        </Text>
                    </Flex>
                </div>
            )}

            {status === 'connecting' && (
                <div className={classes.canvasContainer}>
                    <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                        <Loading mode='inline' />
                    </Flex>
                </div>
            )}

            {status === 'connected' && (
                <div className={classes.canvasContainer}>
                    <Stream size={size} setSize={setSize} />
                </div>
            )}
        </Flex>
    );
};

const ConfigureDisconnectedSource = (_: {
    source: SchemaDisconnectedSourceConfig;
    setSource: (source: SchemaDisconnectedSourceConfig) => void;
}) => {
    return null;
};

const ConfigureImagesFolderSource = ({
    source,
    setSource,
}: {
    source: SchemaImagesFolderSourceConfig;
    setSource: (source: SchemaImagesFolderSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Image folder path'
            name='images_folder_path'
            value={source.images_folder_path}
            onChange={(images_folder_path) => setSource({ ...source, images_folder_path })}
        />
    );
};

const ConfigureIpCameraSource = ({
    source,
    setSource,
}: {
    source: SchemaIpCameraSourceConfig;
    setSource: (source: SchemaIpCameraSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Stream URL'
            name='stream_url'
            value={source.stream_url}
            onChange={(stream_url) => setSource({ ...source, stream_url })}
        />
    );
};

const ConfigureVideoFileSource = ({
    source,
    setSource,
}: {
    source: SchemaVideoFileSourceConfig;
    setSource: (source: SchemaVideoFileSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Video file path'
            name='video_path'
            value={source.video_path}
            onChange={(video_path) => setSource({ ...source, video_path })}
        />
    );
};

const ConfigureWebcamSource = ({
    source,
    setSource,
}: {
    source: SchemaWebcamSourceConfig;
    setSource: (source: SchemaWebcamSourceConfig) => void;
}) => {
    return (
        <NumberField
            label='Webcam device id'
            name='device_id'
            hideStepper
            value={source.device_id}
            onChange={(device_id) => setSource({ ...source, device_id })}
        />
    );
};

const ConfigureSource = ({
    source,
    setSource,
}: {
    source: SourceConfig;
    setSource: (source: SourceConfig) => void;
}) => {
    switch (source.source_type) {
        case 'disconnected':
            return <ConfigureDisconnectedSource source={source} setSource={setSource} />;
        case 'images_folder':
            return <ConfigureImagesFolderSource source={source} setSource={setSource} />;
        case 'ip_camera':
            return <ConfigureIpCameraSource source={source} setSource={setSource} />;
        case 'video_file':
            return <ConfigureVideoFileSource source={source} setSource={setSource} />;
        case 'webcam':
            return <ConfigureWebcamSource source={source} setSource={setSource} />;
    }
};

const Label = ({ item }: { item: { name: string; source_type: SourceType } }) => {
    return (
        <Flex alignItems='center' gap='size-200' margin={0}>
            <Flex alignItems='center' justifyContent={'center'}>
                {item.source_type === 'video_file' && <Video width='32px' />}
                {item.source_type === 'webcam' && <Webcam width='32px' />}
                {item.source_type === 'ip_camera' && <IpCamera width='32px' />}
                {item.source_type === 'images_folder' && <Image width='32px' />}
            </Flex>
            <Text
                UNSAFE_style={{
                    fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                    lineHeight: 'var(--spectrum-global-dimension-size-300)',
                }}
            >
                {item.name}
            </Text>
        </Flex>
    );
};

const DEFAULT_SOURCE_ITEMS = [
    { source_type: 'disconnected', name: 'Disconnected' },
    { source_type: 'webcam', name: 'Webcam' },
    { source_type: 'ip_camera', name: 'IP Camera' },
    { source_type: 'video_file', name: 'Video file' },
    { source_type: 'images_folder', name: 'Images folder' },
] satisfies Array<{ source_type: SourceType; name: string }>;

export const SourceForm = () => {
    const [selectedSourceType, setSelectedSourceType] = useState<SourceType>(DEFAULT_SOURCE_ITEMS[0].source_type);
    const [forms, setForms] = useState<SourceFormRecord>(() => {
        return DEFAULT_SOURCE_FORMS;
    });

    const handleSaveSource = (sourceType: SourceType) => {
        setSelectedSourceType(sourceType);
    };

    const handleSubmit = () => {};

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <RadioDisclosure
                ariaLabel={'Select your source'}
                value={selectedSourceType}
                setValue={setSelectedSourceType}
                items={DEFAULT_SOURCE_ITEMS.map((item) => {
                    return {
                        value: item.source_type,
                        label: <Label item={item} />,
                        content: (
                            <Flex direction={'column'} gap={'size-200'}>
                                <ConfigureSource
                                    source={forms[item.source_type]}
                                    setSource={(newSource) => {
                                        setForms((oldSource) => {
                                            return { ...oldSource, [item.source_type]: newSource };
                                        });
                                    }}
                                />
                                <ButtonGroup>
                                    <Button variant={'accent'} onPress={() => handleSaveSource(item.source_type)}>
                                        Save & connect
                                    </Button>
                                </ButtonGroup>
                            </Flex>
                        ),
                    };
                })}
            />

            <Divider size={'S'} />

            <ButtonGroup alignSelf={'end'}>
                <Button variant={'primary'} onPress={handleSubmit}>
                    Submit
                </Button>
            </ButtonGroup>
        </Flex>
    );
};
