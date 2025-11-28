/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { SchemaProjectList } from '@geti-inspect/api/spec';
import { Flex, PhotoPlaceholder, Text, TextField, type TextFieldRef } from '@geti/ui';
import { clsx } from 'clsx';
import { useNavigate } from 'react-router';

import { useWebRTCConnection } from '../../../../components/stream/web-rtc-connection-provider';
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
        }

        if (e.key === 'Escape') {
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
    isActive: boolean;
    isInEditMode: boolean;
    onBlur: (projectId: string, newName: string) => void;
}

export const ProjectListItem = ({ project, isInEditMode, isActive, onBlur }: ProjectListItemProps) => {
    const navigate = useNavigate();
    const { stop } = useWebRTCConnection();

    const handleBlur = (newProjectId?: string) => (newName: string) => {
        if (newProjectId === undefined) {
            return;
        }

        onBlur(newProjectId, newName);
    };

    const handleNavigateToProject = () => {
        if (project.id === undefined || isActive) {
            return;
        }

        stop();
        navigate(`${paths.project({ projectId: project.id })}?mode=Dataset`);
    };

    return (
        <li
            className={clsx(styles.projectListItem, { [styles.active]: isActive })}
            onClick={isInEditMode ? undefined : handleNavigateToProject}
        >
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
    );
};
