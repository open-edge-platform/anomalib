import { v4 as uuid } from 'uuid';

import { RosSinkConfig, SinkOutputFormats } from '../utils';

export const getRosInitialConfig = (project_id: string): RosSinkConfig => ({
    id: uuid(),
    name: '',
    project_id,
    topic: '',
    rate_limit: 1,
    output_formats: [],
    sink_type: 'ros',
});

const parseRateLimit = (value: FormDataEntryValue | null): number | null => {
    if (!value || value === '') return null;
    return Number(value);
};

export const rosBodyFormatter = (formData: FormData): RosSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    topic: String(formData.get('topic')),
    sink_type: 'ros',
    rate_limit: parseRateLimit(formData.get('rate_limit')),
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
    project_id: String(formData.get('project_id')),
});
