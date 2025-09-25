/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Dataset, Models, Stats } from '@geti-inspect/icons';
import { Flex, Grid, ToggleButton, View } from '@geti/ui';

import styles from './sidebar.module.scss';

const TABS = [
    { label: 'Dataset', icon: <Dataset />, content: <>Dataset</> },
    { label: 'Models', icon: <Models />, content: <>Models</> },
    { label: 'Stats', icon: <Stats />, content: <>Stats</> },
];

interface TabProps {
    tabs: (typeof TABS)[number][];
    selectedTab: string;
}

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);

    const gridTemplateColumns = tab !== null ? ['clamp(size-4600, 35vw, 40rem)', 'size-600'] : ['0px', 'size-600'];

    const content = tabs.find(({ label }) => label === tab)?.content;

    return (
        <Grid
            gridArea={'sidebar'}
            UNSAFE_className={styles.container}
            columns={gridTemplateColumns}
            data-expanded={tab !== null}
        >
            <View
                gridColumn={'1/2'}
                UNSAFE_className={styles.sidebarContent}
                backgroundColor={'gray-100'}
                paddingY={'size-200'}
                paddingX={'size-300'}
            >
                {content}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon }) => (
                        <ToggleButton
                            key={label}
                            isQuiet
                            isSelected={label === tab}
                            onChange={() => setTab(label === tab ? null : label)}
                            UNSAFE_className={styles.toggleButton}
                            aria-label={`Toggle ${label} tab`}
                        >
                            {icon}
                        </ToggleButton>
                    ))}
                </Flex>
            </View>
        </Grid>
    );
};

export const Sidebar = () => {
    return <SidebarTabs tabs={TABS} selectedTab={TABS[0].label} />;
};
