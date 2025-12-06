import { Add as AddIcon } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';
import { Button, Flex, Loading, Text, VirtualizedListLayout } from 'packages/ui';

import { StatusTag } from '../../../../../components/status-tag/status-tag.component';
import { usePipeline } from '../../../../../hooks/use-pipeline.hook';
import { removeUnderscore } from '../../../utils';
import { SourceMenu } from '../source-menu/source-menu.component';
import { SourceConfig } from '../util';
import { SettingsList } from './settings-list/settings-list.component';
import { SourceIcon } from './source-icon/source-icon.component';

import classes from './source-list.module.scss';

type SourcesListProps = {
    sources: SourceConfig[];
    isLoading: boolean;
    onLoadMore: () => void;
    onAddSource: () => void;
    onEditSource: (config: SourceConfig) => void;
};

type SourceListItemProps = {
    source: SourceConfig;
    isConnected: boolean;
    onEditSource: (config: SourceConfig) => void;
};

const SourceListItem = ({ source, isConnected, onEditSource }: SourceListItemProps) => {
    return (
        <Flex
            key={source.id}
            gap='size-200'
            direction='column'
            UNSAFE_className={clsx(classes.card, {
                [classes.activeCard]: isConnected,
            })}
        >
            <Flex alignItems={'center'} gap={'size-200'}>
                <SourceIcon type={source.source_type} />

                <Flex direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.title}>{source.name}</Text>
                    <Flex gap={'size-100'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.type}>{removeUnderscore(source.source_type)}</Text>
                        <StatusTag isConnected={isConnected} />
                    </Flex>
                </Flex>
            </Flex>

            <Flex justifyContent={'space-between'}>
                <SettingsList source={source} />

                <SourceMenu
                    id={String(source.id)}
                    name={source.name}
                    isConnected={isConnected}
                    onEdit={() => onEditSource(source)}
                />
            </Flex>
        </Flex>
    );
};

export const SourcesList = ({ sources, isLoading, onLoadMore, onAddSource, onEditSource }: SourcesListProps) => {
    const pipeline = usePipeline();
    const currentSourceId = pipeline.data.source?.id;

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <Button variant='secondary' UNSAFE_className={classes.addSource} onPress={onAddSource}>
                <AddIcon /> Add new source
            </Button>

            <VirtualizedListLayout
                items={sources}
                isLoading={isLoading}
                onLoadMore={onLoadMore}
                ariaLabel='sources list'
                containerHeight={sources.length > 1 ? 'size-3600' : 'size-3000'}
                layoutOptions={{ gap: 10 }}
                idFormatter={(source: SourceConfig) => String(source.id)}
                textValueFormatter={(source: SourceConfig) => source.name}
                renderLoading={() => <Loading mode={'inline'} size='S' />}
                renderItem={(source: SourceConfig) => (
                    <SourceListItem
                        source={source}
                        isConnected={isEqual(currentSourceId, source.id)}
                        onEditSource={onEditSource}
                    />
                )}
            />
        </Flex>
    );
};
