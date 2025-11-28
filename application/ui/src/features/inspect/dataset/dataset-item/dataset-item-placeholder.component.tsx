import { Image } from '@geti-inspect/icons';
import { Flex } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './dataset-item.module.scss';

export const DatasetItemPlaceholder = () => {
    return (
        <Flex
            justifyContent={'center'}
            alignItems={'center'}
            UNSAFE_className={clsx(styles.datasetItemPlaceholder, styles.datasetItem)}
        >
            <Flex>
                <Image />
            </Flex>
        </Flex>
    );
};
