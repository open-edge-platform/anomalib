/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-inspect/hooks';

import { Project, ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: Project[];
    projectIdInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectsList = ({ projects, setProjectInEdition, projectIdInEdition }: ProjectListProps) => {
    const { projectId: currentProjectId } = useProjectIdentifier();
    const isInEditionMode = (projectId?: string) => {
        return projectIdInEdition === projectId;
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    setProjectInEdition={setProjectInEdition}
                    isInEditMode={isInEditionMode(project.id)}
                    isActive={currentProjectId === project.id}
                />
            ))}
        </ul>
    );
};
