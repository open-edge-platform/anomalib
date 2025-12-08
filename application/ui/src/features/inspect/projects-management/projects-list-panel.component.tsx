/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    PhotoPlaceholder,
    View,
} from '@geti/ui';

import { AddProjectButton } from './add-project-button/add-project-button.component';
import { ProjectsList } from './projects-list.component';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    name: string;
    id: string | undefined;
}

const SelectedProjectButton = ({ name, id }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} staticColor='white'>
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} indicator={id ?? name} height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/projects');

    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);

    const selectedProject = data.projects.find((project) => project.id === projectId);
    const selectedProjectName = selectedProject?.name ?? '';

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} id={selectedProject?.id} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            indicator={selectedProject?.id ?? selectedProjectName}
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content>
                    <Divider size={'S'} marginY={'size-200'} />
                    <ProjectsList
                        projects={data.projects}
                        projectIdInEdition={projectInEdition}
                        setProjectInEdition={setProjectInEdition}
                    />
                    <Divider size={'S'} marginY={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <AddProjectButton
                        onSetProjectInEdition={setProjectInEdition}
                        projectsCount={data.projects.length}
                    />
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
