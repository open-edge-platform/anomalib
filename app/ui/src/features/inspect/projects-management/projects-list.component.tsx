/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { isEmpty } from 'lodash-es';

import { Project, ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: Project[];
    projectIdInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectsList = ({ projects, setProjectInEdition, projectIdInEdition }: ProjectListProps) => {
    /*
    server does not support a project edition
    const updateProjectName = (): void => {
    };*/

    const deleteProject = (): void => {
        // Note: server does not support a project deletion
    };

    const isInEditionMode = (projectId?: string) => {
        return projectIdInEdition === projectId;
    };

    const handleBlur = (projectId: string, newName: string) => {
        setProjectInEdition(null);

        const projectToUpdate = projects.find((project) => project.id === projectId);
        if (projectToUpdate?.name === newName || isEmpty(newName.trim())) {
            return;
        }

        // updateProjectName(projectId, newName);
    };

    /*const handleRename = (projectId: string) => {
        setProjectInEdition(projectId);
    };*/

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    //onRename={handleRename}
                    onDelete={deleteProject}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                />
            ))}
        </ul>
    );
};
