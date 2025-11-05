import { RosSinkConfig, SinkOutputFormats } from '../utils';

export const getRosInitialConfig = (project_id: string): RosSinkConfig => ({
    id: '',
    name: '',
    project_id,
    topic: '',
    rate_limit: 0,
    output_formats: [],
    sink_type: 'ros',
});

export const rosBodyFormatter = (formData: FormData): RosSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    topic: String(formData.get('topic')),
    sink_type: 'ros',
    rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
    project_id: String(formData.get('project_id')),
});
