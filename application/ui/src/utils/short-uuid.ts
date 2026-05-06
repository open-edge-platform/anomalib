/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * ShortUUID generator compatible with Python's `shortuuid` library.
 *
 * Encodes a random UUID (v4) into a 22-character base57 string using
 * the same alphabet as the Python `shortuuid` package.
 *
 * @see {@link https://github.com/skorokithakis/shortuuid}
 */

const ALPHABET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
const BASE = BigInt(ALPHABET.length);
const UUID_LENGTH = 22;

/**
 * Generate a short UUID compatible with the Python `shortuuid` library.
 *
 * @returns A 22-character string using the shortuuid base57 alphabet.
 *
 * @example
 * ```ts
 * generateShortUUID(); // => "iAAFNtgPsJEX2TnLJszCFh"
 * ```
 */
export const generateShortUUID = (): string => {
    const uuid = crypto.randomUUID().replaceAll('-', '');
    let number = BigInt(`0x${uuid}`);

    const chars: string[] = [];
    while (number > 0n) {
        chars.push(ALPHABET[Number(number % BASE)]);
        number = number / BASE;
    }

    while (chars.length < UUID_LENGTH) {
        chars.push(ALPHABET[0]);
    }

    return chars.join('');
};
