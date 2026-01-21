import { removeUnderscore } from '../../../../utils';
import { SinkConfig, SinkOutputFormats, WebhookSinkConfig } from '../../utils';
import { getPairsFromObject } from '../../webhook-fields/utils';

import classes from './settings-list.module.scss';

interface SettingsListProps {
    sink: SinkConfig;
}

const OutputFormats = ({ outputFormats }: { outputFormats: SinkOutputFormats }) => {
    return (
        <ul>
            {outputFormats.map((item) => (
                <li key={item}>{removeUnderscore(item)}</li>
            ))}
        </ul>
    );
};

const WebhookHeaders = ({ sink }: { sink: WebhookSinkConfig }) => {
    return (
        <ul>
            {getPairsFromObject(sink.headers ?? {}).map((pair) => (
                <li key={pair.key}>
                    {pair.key}: {pair.value}
                </li>
            ))}
        </ul>
    );
};

export const SettingsList = ({ sink }: SettingsListProps) => {
    if (sink.sink_type === 'folder') {
        return (
            <ul className={classes.list}>
                <li>Folder path: {sink.folder_path}</li>
                <li>Rate limit: {sink.rate_limit}</li>
                <li>
                    Output formats: <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    if (sink.sink_type === 'webhook') {
        return (
            <ul className={classes.list}>
                <li>Rate limit: {sink.rate_limit}</li>
                <li>HTTP method: {sink.http_method}</li>
                <li>Timeout: {sink.timeout}</li>
                <li>Webhook URL: {sink.webhook_url}</li>
                <li>
                    Headers <WebhookHeaders sink={sink} />
                </li>
                <li>
                    Output formats: <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    if (sink.sink_type === 'mqtt') {
        return (
            <ul className={classes.list}>
                <li>Topic: {sink.topic}</li>
                <li>Rate limit: {sink.rate_limit}</li>
                <li>Auth required: {sink.auth_required ? 'Yes' : 'No'}</li>
                <li>Broker host: {sink.broker_host}</li>
                <li>Broker port: {sink.broker_port}</li>
                <li>
                    Output formats: <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    return <></>;
};
