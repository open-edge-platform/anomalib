import { Button, Flex } from '@geti/ui';
import { Play } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import classes from './play-stream-button.module.scss';

type PlayStreamButtonProps = {
    onStart?: () => void;
    isDisabled?: boolean;
};

export const PlayStreamButton = ({ isDisabled = false, onStart }: PlayStreamButtonProps) => {
    const { data: pipeline } = usePipeline();

    const hasSource = !isEmpty(pipeline?.source);

    return (
        <div className={clsx(classes.container, { [classes.disabled]: isDisabled || !hasSource })} onClick={onStart}>
            <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                <Button
                    onPress={onStart}
                    aria-label={'Start stream'}
                    isDisabled={isDisabled || !hasSource}
                    UNSAFE_className={classes.playButton}
                >
                    <Play width='20px' height='20px' />
                </Button>
            </Flex>
        </div>
    );
};
