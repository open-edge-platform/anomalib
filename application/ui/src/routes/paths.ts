import { path } from 'static-path';

const root = path('/');
const projects = root.path('/projects');
const project = projects.path('/:projectId');
const welcome = path('/welcome');

export const paths = {
    root,
    openapi: root.path('/openapi'),
    project,
    welcome,
};
