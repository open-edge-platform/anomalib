import { ReactNode } from 'react';

import { DisclosureGroup } from 'src/components/disclosure-group/disclosure-group.component';

import { ReactComponent as FolderIcon } from '../../../../assets/icons/folder.svg';
import { ReactComponent as MqttIcon } from '../../../../assets/icons/mqtt.svg';
import { ReactComponent as WebhookIcon } from '../../../../assets/icons/webhook.svg';
import { AddSink } from './add-sink/add-sink.component';
import { LocalFolderFields } from './local-folder-fields/local-folder-fields.component';
import { localFolderBodyFormatter, localFolderInitialConfig } from './local-folder-fields/utils';
import { LocalFolderSinkConfig } from './utils';

interface SinkOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
}

export const SinkOptions = ({ hasHeader, onSaved, children }: SinkOptionsProps) => {
    return (
        <>
            {hasHeader && children}

            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: 'Folder',
                        value: 'folder',
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={localFolderInitialConfig}
                                componentFields={(state: LocalFolderSinkConfig) => <LocalFolderFields state={state} />}
                                bodyFormatter={localFolderBodyFormatter}
                            />
                        ),
                        icon: <FolderIcon width={'24px'} />,
                    },
                    {
                        label: 'Webhook',
                        value: 'webhook',
                        content: <p>Webhook content</p>,
                        icon: <WebhookIcon width={'24px'} />,
                    },
                    {
                        label: 'MQTT',
                        value: 'mqtt',
                        content: <p>MQTT content</p>,
                        icon: <MqttIcon width={'24px'} />,
                    },
                ]}
            />
        </>
    );
};
