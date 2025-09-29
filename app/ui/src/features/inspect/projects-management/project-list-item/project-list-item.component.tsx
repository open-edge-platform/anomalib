/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { SchemaProjectList } from '@geti-inspect/api/spec';
import { AlertDialog, DialogContainer, Flex, PhotoPlaceholder, Text, TextField, type TextFieldRef } from '@geti/ui';
import { useNavigate } from 'react-router';

import { paths } from '../../../../router';

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

interface DeleteProjectDialogProps {
    onDelete: () => void;
    projectName: string;
}

const DeleteProjectDialog = ({ projectName, onDelete }: DeleteProjectDialogProps) => {
    return (
        <AlertDialog
            title='Delete'
            variant='destructive'
            primaryActionLabel='Delete'
            onPrimaryAction={onDelete}
            cancelLabel={'Cancel'}
        >
            {`Are you sure you want to delete project "${projectName}"?`}
        </AlertDialog>
    );
};

/*
Server does not support project actions yet
const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
};

interface ProjectActionsProps {
    onAction: (key: Key) => void;
}

const ProjectActions = ({ onAction }: ProjectActionsProps) => {
    return (
        <ActionMenu isQuiet onAction={onAction} aria-label={'Project actions'} UNSAFE_className={styles.actionMenu}>
            {[PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE].map((action) => (
                <Item key={action}>{action}</Item>
            ))}
        </ActionMenu>
    );
};*/

interface ProjectListItemProps {
    project: Project;
    isInEditMode: boolean;
    onBlur: (projectId: string, newName: string) => void;
    // onRename: (projectId: string) => void;
    onDelete: (projectId: string) => void;
}

export const ProjectListItem = ({ project, isInEditMode, onBlur, onDelete }: ProjectListItemProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const navigate = useNavigate();

    // Server does not support project actions yet
    /*const handleAction = (key: Key) => {
        if (project.id === undefined) {
            return;
        }

        if (key === PROJECT_ACTIONS.RENAME) {
            onRename(project.id);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        }
    };*/

    const handleBlur = (projectId?: string) => (newName: string) => {
        if (projectId === undefined) {
            return;
        }

        onBlur(projectId, newName);
    };

    const handleDelete = () => {
        if (project.id === undefined) {
            return;
        }
        onDelete(project.id);
    };

    const handleNavigateToProject = () => {
        if (project.id === undefined) {
            return;
        }

        navigate(paths.project({ projectId: project.id }));
    };

    return (
        <>
            <li className={styles.projectListItem} onClick={isInEditMode ? undefined : handleNavigateToProject}>
                <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                    {isInEditMode ? (
                        <ProjectEdition name={project.name} onBlur={handleBlur(project.id)} />
                    ) : (
                        <Flex alignItems={'center'} gap={'size-100'}>
                            <PhotoPlaceholder name={project.name} email='' height={'size-300'} width={'size-300'} />
                            <Text>{project.name}</Text>
                        </Flex>
                    )}
                    {/*<ProjectActions onAction={handleAction} />*/}
                </Flex>
            </li>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && <DeleteProjectDialog onDelete={handleDelete} projectName={project.name} />}
            </DialogContainer>
        </>
    );
};
