/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { SchemaProjectList } from '@geti-inspect/api/spec';
import { Flex, PhotoPlaceholder, Text, TextField, type TextFieldRef } from '@geti/ui';
import { useNavigate } from 'react-router';

import { paths } from '../../../../routes/paths';

import styles from './project-list-item.module.scss';

export type Project = SchemaProjectList['projects'][number];

interface ProjectEditionProps {
    onBlur: (newName: string) => void;
    name: string;
}

const ProjectEdition = ({ name, onBlur }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        onBlur(newName);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            onBlur(newName);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setNewName(name);
            onBlur(name);
        }
    };

    useEffect(() => {
        textFieldRef.current?.select();
    }, []);

    return (
        <TextField
            isQuiet
            ref={textFieldRef}
            value={newName}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            onChange={setNewName}
            aria-label='Edit project name'
        />
    );
};

interface ProjectListItemProps {
    project: Project;
    isInEditMode: boolean;
    onBlur: (projectId: string, newName: string) => void;
}

export const ProjectListItem = ({ project, isInEditMode, onBlur }: ProjectListItemProps) => {
    const navigate = useNavigate();

    const handleBlur = (projectId?: string) => (newName: string) => {
        if (projectId === undefined) {
            return;
        }

        onBlur(projectId, newName);
    };

    const handleNavigateToProject = () => {
        if (project.id === undefined) {
            return;
        }

        navigate(`${paths.project({ projectId: project.id })}?mode=Dataset`);
    };

    return (
        <>
            <li className={styles.projectListItem} onClick={isInEditMode ? undefined : handleNavigateToProject}>
                <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                    {isInEditMode ? (
                        <ProjectEdition name={project.name} onBlur={handleBlur(project.id)} />
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
                </Flex>
            </li>
        </>
    );
};
