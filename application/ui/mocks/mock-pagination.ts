import { SchemaPagination } from '../src/api/openapi-spec';

export const getMockedPagination = (overrides?: Partial<SchemaPagination>): SchemaPagination => {
    return {
        offset: 0,
        limit: 0,
        count: 0,
        total: 0,
        ...overrides,
    };
};
