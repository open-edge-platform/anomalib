import { LocalFolderSinkConfig, SinkOutputFormats } from '../utils';

export const localFolderInitialConfig: LocalFolderSinkConfig = {
    id: '',
    name: '',
    sink_type: 'folder',
    rate_limit: 0,
    folder_path: '',
    output_formats: [],
};

export const localFolderBodyFormatter = (formData: FormData): LocalFolderSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    sink_type: 'folder',
    rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
    folder_path: String(formData.get('folder_path')),
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
});
