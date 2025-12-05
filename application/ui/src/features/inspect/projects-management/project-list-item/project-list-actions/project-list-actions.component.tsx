import { Key } from 'react';

import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from 'packages/ui/icons';

interface ProjectActionsProps {
    onRename: () => void;
}

const PROJECT_ACTIONS = {
    RENAME: 'Rename',
};

export const ProjectActions = ({ onRename }: ProjectActionsProps) => {
    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            onRename();
        }
    };
    return (
        <MenuTrigger>
            <ActionButton isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleAction}>
                {[PROJECT_ACTIONS.RENAME].map((action) => (
                    <Item key={action}>{action}</Item>
                ))}
            </Menu>
        </MenuTrigger>
    );
};
