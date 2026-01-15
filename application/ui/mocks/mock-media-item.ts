import { SchemaMediaList } from './../src/api/openapi-spec.d';

export const getMockedMediaItem = (
    data: Partial<SchemaMediaList['media'][number]>
): SchemaMediaList['media'][number] => {
    return {
        id: '1',
        project_id: '123',
        filename: 'test-image.jpg',
        size: 1024,
        is_anomalous: false,
        width: 1920,
        height: 1080,
        created_at: '2024-01-01T00:00:00Z',
        ...data,
    };
};
