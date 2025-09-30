import { Image } from '@geti-inspect/icons';
import { Flex, View } from '@geti/ui';

import { SchemaMediaList } from '../../../../api/openapi-spec';

import styles from './dataset-item.module.scss';

type MediaItem = SchemaMediaList['media'][number];

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
    mediaItem: MediaItem | undefined;
}

export const DatasetItem = ({ mediaItem }: DatasetItemProps) => {
    if (mediaItem === undefined) {
        return <DatasetItemPlaceholder />;
    }

    return <View>Item</View>;
};
