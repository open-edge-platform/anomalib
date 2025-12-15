import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';

const sourcesItemsLimit = 20;

export const useGetSources = () => {
    const { projectId } = useProjectIdentifier();

    const { data, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/sources',
        {
            params: {
                query: { offset: 0, limit: sourcesItemsLimit },
                path: { project_id: projectId },
            },
        },
        {
            pageParamName: 'offset',
            getNextPageParam: ({
                pagination,
            }: {
                pagination: { offset: number; limit: number; count: number; total: number };
            }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + sourcesItemsLimit;
            },
        }
    );

    const sources = data?.pages.flatMap((page) => page.sources) ?? [];

    return { sources, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage };
};
