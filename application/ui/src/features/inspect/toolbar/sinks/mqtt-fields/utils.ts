import { v4 as uuid } from 'uuid';

import { MqttSinkConfig, SinkOutputFormats } from '../utils';

export const getMqttInitialConfig = (project_id: string): MqttSinkConfig => ({
    id: uuid(),
    name: '',
    project_id,
    topic: '',
    sink_type: 'mqtt',
    rate_limit: 0,
    broker_host: '',
    broker_port: 0,
    auth_required: false,
    output_formats: [],
});

export const mqttBodyFormatter = (formData: FormData): MqttSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    topic: String(formData.get('topic')),
    sink_type: 'mqtt',
    rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
    broker_host: String(formData.get('broker_host')),
    broker_port: formData.get('broker_port') ? Number(formData.get('broker_port')) : 0,
    auth_required: formData.get('auth_required') === 'on',
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
    project_id: String(formData.get('project_id')),
});
