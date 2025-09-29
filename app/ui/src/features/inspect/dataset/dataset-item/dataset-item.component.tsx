import { Image } from '@geti-inspect/icons';
import { Flex, View } from '@geti/ui';

import styles from './dataset-item.module.scss';

const DatasetItemPlaceholder = () => {
    return (
        <Flex justifyContent={'center'} alignItems={'center'} UNSAFE_className={styles.datasetItemPlaceholder}>
            <Flex>
                <Image />
            </Flex>
        </Flex>
    );
};

interface DatasetItemProps {
    mediaItem: string | undefined;
}

export const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    if (mediaItem === undefined) {
        return <DatasetItemPlaceholder />;
    }

    return <View>Item</View>;
};
