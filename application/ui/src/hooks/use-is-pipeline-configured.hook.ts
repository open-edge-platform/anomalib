import { isEmpty } from 'lodash-es';
import { SchemaPipeline } from 'src/api/openapi-spec';

export const useIsPipelineConfigured = (pipeline?: SchemaPipeline) => {
    if (!pipeline) return false;

    const { model, source } = pipeline;
    const isEditable = !isEmpty(model) && !isEmpty(source);

    return isEditable;
};
