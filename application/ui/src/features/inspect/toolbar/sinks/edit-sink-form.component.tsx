import { EditSink } from './edit-sink/edit-sink.component';
import { LocalFolderFields } from './local-folder-fields/local-folder-fields.component';
import { localFolderBodyFormatter } from './local-folder-fields/utils';
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
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <LocalFolderFields defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
            />
        );
    }
    if (config.sink_type === 'webhook') {
        return (
            <EditSink
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <WebhookFields defaultState={state} />}
                bodyFormatter={webhookBodyFormatter}
            />
        );
    }

    return <>edith</>;
};
