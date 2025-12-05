import { Flex, NumberField, Switch, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { MqttSinkConfig } from '../utils';

interface MqttFieldsProps {
    defaultState: MqttSinkConfig;
}

export const MqttFields = ({ defaultState }: MqttFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField width='100%' label='Name' name='name' defaultValue={defaultState.name} />
            <TextField width='100%' label='Broker Host' name='broker_host' defaultValue={defaultState.broker_host} />

            <Flex direction='row' gap='size-200'>
                <TextField flex='1' label='Topic' name='topic' defaultValue={defaultState.topic} />
                <NumberField
                    label='Broker Port'
                    name='broker_port'
                    minValue={0}
                    step={1}
                    defaultValue={defaultState.broker_port}
                />
            </Flex>

            <Flex direction='row' gap='size-200' justifyContent='space-between'>
                <NumberField
                    label='Rate Limit'
                    name='rate_limit'
                    minValue={0}
                    step={0.1}
                    defaultValue={defaultState.rate_limit ?? undefined}
                />
                <Switch
                    name='auth_required'
                    alignSelf='end'
                    aria-label='Require Authentication'
                    defaultSelected={defaultState.auth_required}
                    key={defaultState.auth_required ? 'true' : 'false'}
                >
                    Auth Required
                </Switch>
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
