import { SchemaJob } from 'src/api/openapi-spec';

export interface ModelData {
    id: string;
    name: string;
    timestamp: string;
    startTime: number;
    durationInSeconds: number | null;
    status: 'Training' | 'Completed' | 'Failed';
    architecture: string;
    progress: number;
    job: SchemaJob | undefined;
    sizeBytes: number | null;
    backbone: string | null;
}
