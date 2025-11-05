import { getObjectFromFormData } from './utils';

describe('getObjectFromFormData', () => {
    it('return an object mapping keys to values', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ a: '1', b: '2', c: '3' });
    });

    it('skip entries with empty keys', () => {
        const keys = ['', 'b', ''];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ b: '2' });
    });

    it('skip entries with empty values', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['', '2', ''];
        expect(getObjectFromFormData(keys, values)).toEqual({ b: '2' });
    });

    it('return an empty object if all keys are empty', () => {
        const keys = ['', '', ''];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({});
    });

    it('return an empty object if all values are empty', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['', '', ''];
        expect(getObjectFromFormData(keys, values)).toEqual({});
    });

    it('handle different lengths of keys and values', () => {
        const keys = ['a', 'b'];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ a: '1', b: '2' });
    });

    it('handle keys and values with whitespace', () => {
        const keys = [' ', 'b', 'c'];
        const values = ['1', ' ', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ c: '3' });
    });
});
