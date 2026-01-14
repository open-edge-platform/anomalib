// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export interface LogTime {
    timestamp: number;
    repr: string;
}

export interface LogLevel {
    name: LogLevelName;
    no: number;
    icon: string;
}

export type LogLevelName = 'TRACE' | 'DEBUG' | 'INFO' | 'SUCCESS' | 'WARNING' | 'ERROR' | 'CRITICAL';

export const LOG_LEVELS: LogLevelName[] = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'];

// Loguru-inspired color palette
export const LOG_LEVEL_COLORS: Record<LogLevelName, string> = {
    TRACE: '#06b6d4',    // Cyan
    DEBUG: '#6b7280',    // Gray (swapped with INFO)
    INFO: '#3b82f6',     // Blue (swapped with DEBUG)
    SUCCESS: '#22c55e',  // Green
    WARNING: '#eab308',  // Yellow
    ERROR: '#ef4444',    // Red
    CRITICAL: '#dc2626', // Red with special styling
};

export interface LogProcess {
    id: number;
    name: string;
}

export interface LogThread {
    id: number;
    name: string;
}

export interface LogFile {
    name: string;
    path: string;
}

export interface LogRecord {
    elapsed: { repr: string; seconds: number };
    exception: unknown;
    extra: Record<string, unknown>;
    file: LogFile;
    function: string;
    level: LogLevel;
    line: number;
    message: string;
    module: string;
    name: string;
    process: LogProcess;
    thread: LogThread;
    time: LogTime;
}

export interface LogEntry {
    text: string;
    record: LogRecord;
}

export interface LogFilters {
    levels: Set<LogLevelName>;
    searchQuery: string;
    startTime: number | null;
    endTime: number | null;
}

export const DEFAULT_LOG_FILTERS: LogFilters = {
    levels: new Set(LOG_LEVELS),
    searchQuery: '',
    startTime: null,
    endTime: null,
};

