// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from 'src/api/openapi-spec';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfig-Input'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfig-Input'];
export type WebcamSourceConfig = components['schemas']['WebcamSourceConfig-Input'];
export type VideoFileSourceConfig = components['schemas']['VideoFileSourceConfig-Input'];

export type SourceConfig = ImagesFolderSourceConfig | IPCameraSourceConfig | WebcamSourceConfig | VideoFileSourceConfig;

export const isOnlyDigits = (str: string): boolean => {
    return /^\d+$/.test(str);
};

export const getImageFolderData = <T extends { source_type: string }>(sources: T[]) => {
    return sources
        .filter(({ source_type }) => source_type === 'images_folder')
        .at(0) as unknown as ImagesFolderSourceConfig;
};

export const getIpCameraData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'ip_camera').at(0) as unknown as IPCameraSourceConfig;
};

export const getWebcamData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'webcam').at(0) as unknown as WebcamSourceConfig;
};

export const getVideoFileData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'video_file').at(0) as unknown as VideoFileSourceConfig;
};
