/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-inspect/api';
import { SchemaProjectList } from '@geti-inspect/api/spec';
import { Flex, PhotoPlaceholder, Text } from '@geti/ui';
import { clsx } from 'clsx';
import { useNavigate } from 'react-router';

import { paths } from '../../../../routes/paths';
import { ProjectEdition } from './project-edition/project-edition.component';
import { ProjectActions } from './project-list-actions/project-list-actions.component';

import styles from './project-list-item.module.scss';

export type Project = SchemaProjectList['projects'][number];

interface ProjectListItemProps {
    project: Project;
    isActive: boolean;
    isInEditMode: boolean;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectListItem = ({ project, isActive, isInEditMode, setProjectInEdition }: ProjectListItemProps) => {
    const navigate = useNavigate();

    const updateProject = $api.useMutation('patch', '/api/projects/{project_id}', {
        onSettled: () => setProjectInEdition(null),
        meta: { invalidates: [['get', '/api/projects']] },
    });

    const handleNameChange = (projectId?: string) => (newName: string) => {
        if (projectId === undefined) {
            return;
        }

        updateProject.mutate({
            params: { path: { project_id: projectId } },
            body: { name: newName },
        });
    };

    const handleNavigateToProject = () => {
        if (project.id === undefined || isActive) {
            return;
        }

        navigate(`${paths.project({ projectId: project.id })}?mode=Dataset`);
    };

    return (
        <li
            className={clsx(styles.projectListItem, { [styles.active]: isActive })}
            onClick={isInEditMode ? undefined : handleNavigateToProject}
        >
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                {isInEditMode ? (
                    <ProjectEdition
                        name={project.name}
                        onChange={handleNameChange(project.id)}
                        isPending={updateProject.isPending}
                    />
                ) : (
                    <Flex alignItems={'center'} gap={'size-100'}>
                        <PhotoPlaceholder
                            name={project.name}
                            indicator={project.id ?? project.name}
                            height={'size-300'}
                            width={'size-300'}
                        />
                        <Text>{project.name}</Text>
                    </Flex>
                )}

                <ProjectActions onRename={() => setProjectInEdition(project.id ?? null)} />
            </Flex>
        </li>
    );
};
