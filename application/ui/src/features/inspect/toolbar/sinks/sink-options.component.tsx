import { ReactNode } from 'react';

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Folder as FolderIcon, Mqtt as MqttIcon, Webhook as WebhookIcon } from '@geti-inspect/icons';

import { DisclosureGroup } from '../../../../components/disclosure-group/disclosure-group.component';
import { AddSink } from './add-sink/add-sink.component';
import { LocalFolderFields } from './local-folder-fields/local-folder-fields.component';
import { getLocalFolderInitialConfig, localFolderBodyFormatter } from './local-folder-fields/utils';
import { MqttFields } from './mqtt-fields/mqtt-fields.component';
import { getMqttInitialConfig, mqttBodyFormatter } from './mqtt-fields/utils';
import { LocalFolderSinkConfig, MqttSinkConfig, WebhookSinkConfig } from './utils';
import { getWebhookInitialConfig, webhookBodyFormatter } from './webhook-fields/utils';
import { WebhookFields } from './webhook-fields/webhook-fields.component';

interface SinkOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
}

export const SinkOptions = ({ hasHeader, onSaved, children }: SinkOptionsProps) => {
    const { projectId } = useProjectIdentifier();

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
                                config={getLocalFolderInitialConfig(projectId)}
                                componentFields={(state: LocalFolderSinkConfig) => (
                                    <LocalFolderFields defaultState={state} />
                                )}
                                bodyFormatter={localFolderBodyFormatter}
                            />
                        ),
                        icon: <FolderIcon width={'24px'} />,
                    },
                    {
                        label: 'Webhook',
                        value: 'webhook',
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={getWebhookInitialConfig(projectId)}
                                componentFields={(state: WebhookSinkConfig) => <WebhookFields defaultState={state} />}
                                bodyFormatter={webhookBodyFormatter}
                            />
                        ),
                        icon: <WebhookIcon width={'24px'} />,
                    },
                    {
                        label: 'MQTT',
                        value: 'mqtt',
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={getMqttInitialConfig(projectId)}
                                componentFields={(state: MqttSinkConfig) => <MqttFields defaultState={state} />}
                                bodyFormatter={mqttBodyFormatter}
                            />
                        ),
                        icon: <MqttIcon width={'24px'} />,
                    },
                ]}
            />
        </>
    );
};
