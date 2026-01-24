import { Folder, Mqtt, Webhook } from '@geti-inspect/icons';

interface SinkIconProps {
    type: 'folder' | 'mqtt' | 'webhook' | 'disconnected';
}

export const SinkIcon = ({ type }: SinkIconProps) => {
    if (type === 'folder') {
        return <Folder />;
    }

    if (type === 'mqtt') {
        return <Mqtt />;
    }

    if (type === 'webhook') {
        return <Webhook />;
    }

    return <></>;
};
