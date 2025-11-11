import { Suspense } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Icon,
    Loading,
    Text,
    View,
} from '@geti/ui';
import { LogsIcon } from '@geti/ui/icons';
import { queryOptions, experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';
import { fetchSSE } from 'src/api/fetch-sse';

const JobLogsDialogContent = ({ jobId }: { jobId: string }) => {
    const query = useQuery(
        queryOptions({
            queryKey: ['get', '/api/jobs/{job_id}/logs', jobId],
            queryFn: streamedQuery({
                queryFn: () => fetchSSE(`/api/jobs/${jobId}/logs`),
            }),
            staleTime: Infinity,
        })
    );

    return (
        <Flex direction='column' gap='size-25'>
            {query.data?.map((line, idx) => <Text key={idx}> {line.text}</Text>)}
        </Flex>
    );
};

const JobLogsDialog = ({ close, jobId }: { close: () => void; jobId: string }) => {
    return (
        <Dialog>
            <Heading>Logs</Heading>
            <Divider />
            <Content>
                <View
                    padding='size-200'
                    backgroundColor={'gray-50'}
                    UNSAFE_style={{
                        fontSize: 'var(--spectrum-global-dimension-static-size-130)',
                    }}
                >
                    <Suspense fallback={<Loading mode='inline' />}>
                        <JobLogsDialogContent jobId={jobId} />
                    </Suspense>
                </View>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const ShowJobLogs = ({ jobId }: { jobId: string }) => {
    return (
        <View>
            <DialogTrigger type='fullscreen'>
                <ActionButton>
                    <Icon>
                        <LogsIcon />
                    </Icon>
                </ActionButton>
                {(close) => <JobLogsDialog close={close} jobId={jobId} />}
            </DialogTrigger>
        </View>
    );
};
