// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';
import { components } from 'src/api/openapi-spec';

export type LocalFolderSinkConfig = components['schemas']['FolderSinkConfig-Output'];
export type MqttSinkConfig = components['schemas']['MqttSinkConfig-Output'];
export type WebhookSinkConfig = components['schemas']['WebhookSinkConfig-Output'];
export type RosSinkConfig = components['schemas']['RosSinkConfig-Output'];
export type SinkOutputFormats = LocalFolderSinkConfig['output_formats'];

export type SinkConfig = LocalFolderSinkConfig | MqttSinkConfig | WebhookSinkConfig | RosSinkConfig;

export enum SinkType {
    FOLDER = 'folder',
    MQTT = 'mqtt',
    ROS = 'ros',
    WEBHOOK = 'webhook',
}

export enum OutputFormat {
    IMAGE_ORIGINAL = 'image_original',
    IMAGE_WITH_PREDICTIONS = 'image_with_predictions',
    PREDICTIONS = 'predictions',
}

export enum WebhookHttpMethod {
    PUT = 'PUT',
    POST = 'POST',
    PATCH = 'PATCH',
}

const toStringAndTrim = (value: unknown) => String(value).trim();

export const getObjectFromFormData = (keys: FormDataEntryValue[], values: FormDataEntryValue[]) => {
    const entries = keys.map((key, index) => [key, values[index]]);
    const validEntries = entries.filter(
        ([key, value]) => !isEmpty(toStringAndTrim(key)) && !isEmpty(toStringAndTrim(value))
    );

    return Object.fromEntries(validEntries);
};

export const getLocalFolderData = <T extends { sink_type: string }>(sources: T[]) => {
    return sources.filter(({ sink_type }) => sink_type === 'folder').at(0) as unknown as LocalFolderSinkConfig;
};

export const getMqttData = <T extends { sink_type: string }>(sources: T[]) => {
    return sources.filter(({ sink_type }) => sink_type === 'mqtt').at(0) as unknown as MqttSinkConfig;
};

export const getWebhookData = <T extends { sink_type: string }>(sources: T[]) => {
    return sources.filter(({ sink_type }) => sink_type === 'webhook').at(0) as unknown as WebhookSinkConfig;
};
