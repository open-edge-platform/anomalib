import { removeUnderscore } from 'src/features/utils';

import { SinkConfig } from '../../utils';

import classes from './settings-list.module.scss';

interface SettingsListProps {
    source: SinkConfig;
}

export const SettingsList = ({ source }: SettingsListProps) => {
    if (source.sink_type === 'folder') {
        return (
            <ul className={classes.list}>
                <li>Folder path: {source.folder_path}</li>
                <li>Rate limit: {source.rate_limit}</li>
                <li>Output formats: {source.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    if (source.sink_type === 'webhook') {
        return (
            <ul className={classes.list}>
                <li>Output formats: {source.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    if (source.sink_type === 'mqtt') {
        return (
            <ul className={classes.list}>
                <li>Topic: {source.topic}</li>
                <li>Rate limit: {source.rate_limit}</li>
                <li>Auth required: {source.auth_required ? 'Yes' : 'No'}</li>
                <li>Broker host: {source.broker_host}</li>
                <li>Broker port: {source.broker_port}</li>
                <li>Output formats: {source.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    if (source.sink_type === 'ros') {
        return (
            <ul className={classes.list}>
                <li>Topic: {source.topic}</li>
                <li>Rate limit: {source.rate_limit}</li>
                <li>Output formats: {source.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    return <></>;
};
