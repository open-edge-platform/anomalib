import type { SchemaProjectList } from '@geti-inspect/api/spec';

export const getMockedProject = (
    partial?: Partial<SchemaProjectList['projects'][number]>
): SchemaProjectList['projects'][number] => ({
    id: 'project-1',
    name: 'Test Project',
    ...partial,
});
