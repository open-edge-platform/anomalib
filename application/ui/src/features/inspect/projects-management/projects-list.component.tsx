/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { LoadMoreList } from 'src/components/load-more-list/load-more-list.component';

import { Project, ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: Project[];
    projectIdInEdition: string | null;
    isLoading: boolean;
    hasNextPage: boolean;
    onLoadMore: () => void;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectsList = ({
    projects,
    isLoading,
    hasNextPage,
    projectIdInEdition,
    onLoadMore,
    setProjectInEdition,
}: ProjectListProps) => {
    const { projectId: currentProjectId } = useProjectIdentifier();
    const isInEditionMode = (projectId?: string) => {
        return projectIdInEdition === projectId;
    };

    const isLastProject = projects.length <= 1;

    return (
        <LoadMoreList isLoading={isLoading} hasNextPage={hasNextPage} onLoadMore={onLoadMore}>
            <ul className={styles.projectList}>
                {projects.map((project) => (
                    <ProjectListItem
                        key={project.id}
                        project={project}
                        setProjectInEdition={setProjectInEdition}
                        isInEditMode={isInEditionMode(project.id)}
                        isActive={currentProjectId === project.id}
                        isLastProject={isLastProject}
                    />
                ))}
            </ul>
        </LoadMoreList>
    );
};
