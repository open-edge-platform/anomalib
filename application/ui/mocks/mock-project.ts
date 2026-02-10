import type { SchemaProjectList } from '@anomalib-studio/api/spec';

export const getMockedProject = (
    partial?: Partial<SchemaProjectList['projects'][number]>
): SchemaProjectList['projects'][number] => ({
    id: 'project-1',
    name: 'Test Project',
    ...partial,
});
