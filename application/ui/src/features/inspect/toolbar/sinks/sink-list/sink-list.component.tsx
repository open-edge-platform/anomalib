import { Add as AddIcon } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';
import { Button, Flex, Text } from 'packages/ui';
import { removeUnderscore } from 'src/features/utils';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { SinkConfig } from '../utils';
import { SettingsList } from './settings-list/settings-list.component';
import { SinkMenu } from './sink-menu/sink-menu.component';
import { SourceIcon } from './source-icon/source-icon.component';
import { StatusTag } from './status-tag/status-tag.component';

import classes from './sink-list.module.scss';

type SinksListProps = {
    sinks: SinkConfig[];
    onAddSink: () => void;
    onEditSourceFactory: (config: SinkConfig) => void;
};

export const SinkList = ({ sinks, onAddSink, onEditSourceFactory }: SinksListProps) => {
    const pipeline = usePipeline();
    const currentSource = pipeline.data.source?.id;

    return (
        <Flex gap={'size-200'} direction={'column'}>
            <Button variant='secondary' UNSAFE_className={classes.addSink} onPress={onAddSink}>
                <AddIcon /> Add new sink
            </Button>

            {sinks.map((sink) => {
                const isConnected = isEqual(currentSource, sink.id);

                return (
                    <Flex
                        key={sink.id}
                        gap='size-200'
                        direction='column'
                        UNSAFE_className={clsx(classes.card, {
                            [classes.activeCard]: isConnected,
                        })}
                    >
                        <Flex alignItems={'center'} gap={'size-200'}>
                            <SourceIcon type={sink.sink_type} />

                            <Flex direction={'column'} gap={'size-100'}>
                                <Text UNSAFE_className={classes.title}>{sink.name}</Text>
                                <Flex gap={'size-100'} alignItems={'center'}>
                                    <Text UNSAFE_className={classes.type}>{removeUnderscore(sink.sink_type)}</Text>
                                    <StatusTag isConnected={isConnected} />
                                </Flex>
                            </Flex>
                        </Flex>

                        <Flex justifyContent={'space-between'}>
                            <SettingsList source={sink} />

                            <SinkMenu id={String(sink.id)} name={sink.name} onEdit={() => onEditSourceFactory(sink)} />
                        </Flex>
                    </Flex>
                );
            })}
        </Flex>
    );
};
