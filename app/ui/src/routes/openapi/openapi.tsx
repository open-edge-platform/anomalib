import { Flex, Header as SpectrumHeader, View } from '@geti/ui';
import { ApiReferenceReact } from '@scalar/api-reference-react';
import { Link } from 'react-router-dom';

import classes from './openapi.module.scss';

import '@scalar/api-reference-react/style.css';

const Header = () => {
    return (
        <SpectrumHeader UNSAFE_className={classes.header}>
            <View padding={'size-200'}>
                <Link to={'/'}>Back</Link>
            </View>
        </SpectrumHeader>
    );
};

export const OpenApi = () => {
    return (
        <Flex direction={'column'} UNSAFE_className={classes.container} height={'100vh'}>
            <Header />
            <View flex={1} minHeight={0} overflow={'hidden auto'}>
                <ApiReferenceReact
                    configuration={{
                        url: '/api/openapi.json',
                        layout: 'modern',
                        showSidebar: true,
                        hideModels: true,
                        hideClientButton: true,
                        hideDarkModeToggle: true,
                        metaData: {
                            title: 'Geti Action | REST API specification',
                        },
                        servers: [{ url: `/api/`, description: 'Geti Action' }],
                        forceDarkModeState: 'dark',
                    }}
                />
            </View>
        </Flex>
    );
};
