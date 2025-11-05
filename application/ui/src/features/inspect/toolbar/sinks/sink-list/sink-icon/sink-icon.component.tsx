import { Folder, Mqtt, Ros, Webhook } from '@geti-inspect/icons';

interface SinkIconProps {
    type: 'folder' | 'mqtt' | 'webhook' | 'ros' | 'disconnected';
}

export const SinkIcon = ({ type }: SinkIconProps) => {
    if (type === 'folder') {
        return <Folder />;
    }

    if (type === 'mqtt') {
        return <Mqtt />;
    }

    if (type === 'ros') {
        return <Ros />;
    }

    if (type === 'webhook') {
        return <Webhook />;
    }

    return <></>;
};
