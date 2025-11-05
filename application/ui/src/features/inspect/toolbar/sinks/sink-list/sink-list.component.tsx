import { Add as AddIcon } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';
import { Button, Flex, Text } from 'packages/ui';
import { removeUnderscore } from 'src/features/utils';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { SinkConfig } from '../utils';
import { SettingsList } from './settings-list/settings-list.component';
import { SinkIcon } from './sink-icon/sink-icon.component';
import { SinkMenu } from './sink-menu/sink-menu.component';
import { StatusTag } from './status-tag/status-tag.component';

import classes from './sink-list.module.scss';

type SinksListProps = {
    sinks: SinkConfig[];
    onAddSink: () => void;
    onEditSink: (config: SinkConfig) => void;
};

export const SinkList = ({ sinks, onAddSink, onEditSink }: SinksListProps) => {
    const pipeline = usePipeline();
    const currentSink = pipeline.data.sink?.id;

    return (
        <Flex gap={'size-200'} direction={'column'}>
            <Button variant='secondary' UNSAFE_className={classes.addSink} onPress={onAddSink}>
                <AddIcon /> Add new sink
            </Button>

            {sinks.map((sink) => {
                const isConnected = isEqual(currentSink, sink.id);

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
                            <SinkIcon type={sink.sink_type} />

                            <Flex direction={'column'} gap={'size-100'}>
                                <Text UNSAFE_className={classes.title}>{sink.name}</Text>
                                <Flex gap={'size-100'} alignItems={'center'}>
                                    <Text UNSAFE_className={classes.type}>{removeUnderscore(sink.sink_type)}</Text>
                                    <StatusTag isConnected={isConnected} />
                                </Flex>
                            </Flex>
                        </Flex>

                        <Flex justifyContent={'space-between'}>
                            <SettingsList sink={sink} />

                            <SinkMenu id={String(sink.id)} name={sink.name} onEdit={() => onEditSink(sink)} />
                        </Flex>
                    </Flex>
                );
            })}
        </Flex>
    );
};
