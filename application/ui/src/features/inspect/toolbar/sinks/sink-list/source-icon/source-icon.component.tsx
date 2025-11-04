import { Folder, Mqtt, Ros, Webhook } from '@geti-inspect/icons';

interface SourceIconProps {
    type: 'folder' | 'mqtt' | 'ros' | 'webhook';
}

export const SourceIcon = ({ type }: SourceIconProps) => {
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
