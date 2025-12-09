import { ReactNode } from 'react';

import { Button, Flex } from '@geti/ui';
import { useListEnd } from 'src/hooks/use-list-end.hook';

type LoadMoreListProps = {
    children: ReactNode;
    isLoading: boolean;
    hasNextPage: boolean;
    onLoadMore: () => void;
};

export const LoadMoreList = ({ children, isLoading, hasNextPage, onLoadMore }: LoadMoreListProps) => {
    const sentinelRef = useListEnd({ onEndReached: onLoadMore, disabled: isLoading });

    return (
        <Flex gap={'size-200'} direction={'column'} maxHeight={'60vh'} UNSAFE_style={{ overflow: 'auto' }}>
            {children}

            {hasNextPage && (
                <Button ref={sentinelRef} isDisabled={isLoading} isPending={isLoading} onPress={onLoadMore}>
                    Load More
                </Button>
            )}
        </Flex>
    );
};
