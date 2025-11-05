import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { ActionButton, Flex, Text } from '@geti/ui';
import { Back } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { EditSinkForm } from './edit-sink-form.component';
import { SinkList } from './sink-list/sink-list.component';
import { SinkOptions } from './sink-options.component';
import { SinkConfig } from './utils';

export const SinkActions = () => {
    const { projectId } = useProjectIdentifier();
    const sinksQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/sinks', {
        params: { path: { project_id: projectId } },
    });

    const sinks = sinksQuery.data ?? [];

    const [view, setView] = useState<'list' | 'options' | 'edit'>(isEmpty(sinks) ? 'options' : 'list');
    const [currentSink, setCurrentSink] = useState<SinkConfig | null>(null);

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSinks = () => {
        setView('options');
    };

    const handleEditSink = (sink: SinkConfig) => {
        setView('edit');
        setCurrentSink(sink);
    };

    if (view === 'edit' && !isEmpty(currentSink)) {
        return <EditSinkForm config={currentSink} onSaved={handleShowList} onBackToList={handleShowList} />;
    }

    if (view === 'list') {
        return <SinkList sinks={sinks} onAddSink={handleAddSinks} onEditSink={handleEditSink} />;
    }

    return (
        <SinkOptions onSaved={handleShowList} hasHeader={sinks.length > 0}>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>Add new sink</Text>
            </Flex>
        </SinkOptions>
    );
};
