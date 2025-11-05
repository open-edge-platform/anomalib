import { removeUnderscore } from 'src/features/utils';

import { SinkConfig } from '../../utils';

import classes from './settings-list.module.scss';

interface SettingsListProps {
    sink: SinkConfig;
}

export const SettingsList = ({ sink }: SettingsListProps) => {
    if (sink.sink_type === 'folder') {
        return (
            <ul className={classes.list}>
                <li>Folder path: {sink.folder_path}</li>
                <li>Rate limit: {sink.rate_limit}</li>
                <li>Output formats: {sink.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    if (sink.sink_type === 'webhook') {
        return (
            <ul className={classes.list}>
                <li>Output formats: {sink.output_formats.map(removeUnderscore).join(', ')}</li>
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
                <li>Output formats: {sink.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    if (sink.sink_type === 'ros') {
        return (
            <ul className={classes.list}>
                <li>Topic: {sink.topic}</li>
                <li>Rate limit: {sink.rate_limit}</li>
                <li>Output formats: {sink.output_formats.map(removeUnderscore).join(', ')}</li>
            </ul>
        );
    }

    return <></>;
};
