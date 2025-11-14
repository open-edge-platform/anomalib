/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-inspect/hooks';
import { isEmpty } from 'lodash-es';

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

    const handleBlur = (projectId: string, newName: string) => {
        setProjectInEdition(null);

        const projectToUpdate = projects.find((project) => project.id === projectId);
        if (projectToUpdate?.name === newName || isEmpty(newName.trim())) {
            return;
        }
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                    isActive={currentProjectId === project.id}
                />
            ))}
        </ul>
    );
};
