import { Add as AddIcon } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';
import { Button, Flex, Text } from 'packages/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { SourceMenu } from '../source-menu/source-menu.component';
import { SourceConfig } from '../util';
import { SettingsList } from './settings-list/settings-list.component';
import { SourceIcon } from './source-icon/source-icon.component';
import { StatusTag } from './status-tag/status-tag.component';
import { removeUnderscore } from './utils';

import classes from './source-list.module.scss';

type SourcesListProps = {
    sources: SourceConfig[];
    onAddSource: () => void;
    onEditSourceFactory: (config: SourceConfig) => void;
};

export const SourcesList = ({ sources, onAddSource, onEditSourceFactory }: SourcesListProps) => {
    const pipeline = usePipeline();
    const currentSource = pipeline.data.source?.id;

    return (
        <Flex gap={'size-200'} direction={'column'}>
            <Button variant='secondary' UNSAFE_className={classes.addSource} onPress={onAddSource}>
                <AddIcon /> Add new source
            </Button>

            {sources.map((source) => {
                const isConnected = isEqual(currentSource, source.id);

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
                                onEdit={() => onEditSourceFactory(source)}
                            />
                        </Flex>
                    </Flex>
                );
            })}
        </Flex>
    );
};
