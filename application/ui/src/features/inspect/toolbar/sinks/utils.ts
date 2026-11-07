import { isEmpty } from 'lodash-es';
import { components } from 'src/api/openapi-spec';

export type LocalFolderSinkConfig = components['schemas']['FolderSinkConfig'];
export type MqttSinkConfig = components['schemas']['MqttSinkConfig'];
export type WebhookSinkConfig = components['schemas']['WebhookSinkConfig'];
export type DisconnectedSinkConfig = components['schemas']['DisconnectedSinkConfig'];

export type SinkOutputFormats = LocalFolderSinkConfig['output_formats'];

export type SinkConfig = LocalFolderSinkConfig | MqttSinkConfig | WebhookSinkConfig | DisconnectedSinkConfig;

export enum SinkType {
    FOLDER = 'folder',
    MQTT = 'mqtt',
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
