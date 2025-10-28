import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { MoreMenu } from '@geti/ui/icons';
import { ActionButton, Item, Key, Menu, MenuTrigger, toast } from 'packages/ui';

interface SourceMenuProps {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
}

export const SourceMenu = ({ id, name, isConnected, onEdit }: SourceMenuProps) => {
    const { projectId } = useProjectIdentifier();

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/sources'],
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const removeSource = $api.useMutation('delete', '/api/sources/{source_id}', {
        meta: {
            invalidates: [['get', '/api/sources']],
        },
    });

    const handleOnAction = (option: Key) => {
        switch (option) {
            case 'connect':
                handleConnect();
                break;
            case 'remove':
                handleDelete();
                break;
            default:
                onEdit();
                break;
        }
    };

    const handleConnect = async () => {
        await updatePipeline.mutateAsync({
            params: { path: { project_id: projectId } },
            body: { source_id: id },
        });

        toast({
            type: 'success',
            message: `Successfully connected to "${name}"`,
        });
    };

    const handleDelete = async () => {
        if (isConnected) {
            await updatePipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { source_id: null },
            });
        }

        await removeSource.mutateAsync({ params: { path: { source_id: id } } });

        toast({
            type: 'success',
            message: `${name} has been removed successfully!`,
        });
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleOnAction}>
                <Item key='connect'>Connect</Item>
                <Item key='edit'>Edit</Item>
                <Item key='remove'>Remove</Item>
            </Menu>
        </MenuTrigger>
    );
};
