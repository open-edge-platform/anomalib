import { EditSink } from './edit-sink/edit-sink.component';
import { LocalFolderFields } from './local-folder-fields/local-folder-fields.component';
import { localFolderBodyFormatter } from './local-folder-fields/utils';
import { MqttFields } from './mqtt-fields/mqtt-fields.component';
import { mqttBodyFormatter } from './mqtt-fields/utils';
import { SinkConfig } from './utils';
import { webhookBodyFormatter } from './webhook-fields/utils';
import { WebhookFields } from './webhook-fields/webhook-fields.component';

interface EditSinkFormProps {
    config: SinkConfig;
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSinkForm = ({ config, onSaved, onBackToList }: EditSinkFormProps) => {
    if (config.sink_type === 'folder') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <LocalFolderFields defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
            />
        );
    }

    if (config.sink_type === 'webhook') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <WebhookFields defaultState={state} />}
                bodyFormatter={webhookBodyFormatter}
            />
        );
    }

    if (config.sink_type === 'mqtt') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <MqttFields defaultState={state} />}
                bodyFormatter={mqttBodyFormatter}
            />
        );
    }

    return <></>;
};
