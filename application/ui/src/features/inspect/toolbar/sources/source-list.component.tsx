import { Add as AddIcon } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';
import { Button, Flex, Text } from 'packages/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { ReactComponent as IpCameraIcon } from '../../../../assets/icons/ip-camera.svg';
import { SourceMenu } from './source-menu/source-menu.component';
import { StatusTag } from './status-tag/status-tag.component';
import { SourceConfig } from './util';

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

            {sources.map((source) => (
                <Flex
                    key={source.id}
                    gap='size-200'
                    direction='column'
                    UNSAFE_className={clsx(classes.card, classes.activeCard)}
                >
                    <Flex alignItems={'center'} gap={'size-200'}>
                        <IpCameraIcon />

                        <Flex direction={'column'} gap={'size-100'}>
                            <Text UNSAFE_className={classes.title}>{source.name}</Text>
                            <Flex gap={'size-100'} alignItems={'center'}>
                                <Text>{source.source_type}</Text>
                                <StatusTag isConnected={isEqual(currentSource, source.id)} />
                            </Flex>
                        </Flex>
                    </Flex>

                    <Flex justifyContent={'space-between'}>
                        <ul className={classes.list}>
                            <li>device_id: 123</li>
                        </ul>

                        <SourceMenu
                            id={String(source.id)}
                            name={source.name}
                            isConnected={isEqual(currentSource, source.id)}
                            onEdit={() => onEditSourceFactory(source)}
                        />
                    </Flex>
                </Flex>
            ))}
        </Flex>
    );
};
