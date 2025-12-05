import { SchemaModel } from '../src/api/openapi-spec';

export const getMockedModel = (partial: Partial<SchemaModel>): SchemaModel => ({
    id: 'model-1',
    name: 'Model 1',
    format: 'openvino' as const,
    project_id: '123',
    threshold: 0.5,
    is_ready: true,
    dataset_snapshot_id: 'dataset-1',
    train_job_id: 'job-1',
    ...partial,
});
