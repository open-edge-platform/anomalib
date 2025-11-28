// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from 'src/api/openapi-spec';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfig'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfig'];
export type WebcamSourceConfig = components['schemas']['WebcamSourceConfig'];
export type VideoFileSourceConfig = components['schemas']['VideoFileSourceConfig'];
export type DisconnectedSourceConfig = components['schemas']['DisconnectedSourceConfig'];

export type SourceConfig =
    | ImagesFolderSourceConfig
    | IPCameraSourceConfig
    | WebcamSourceConfig
    | VideoFileSourceConfig
    | DisconnectedSourceConfig;

export type SourceTypes = SourceConfig['source_type'];

export const isOnlyDigits = (str: string): boolean => {
    return /^\d+$/.test(str);
};
