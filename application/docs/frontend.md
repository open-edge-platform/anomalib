# Frontend Documentation

The Geti Inspect frontend is a modern React application built with TypeScript, providing an intuitive interface for managing anomaly detection projects, training models, and running inference pipelines.

## Architecture Overview

```
ui/
├── src/
│   ├── api/                    # API client layer
│   │   ├── openapi-spec.json   # Generated OpenAPI spec
│   │   ├── openapi-spec.d.ts   # Generated TypeScript types
│   │   └── client.ts           # API client setup
│   ├── components/             # Reusable components
│   ├── hooks/                  # Custom React hooks
│   ├── pages/                  # Page components
│   ├── styles/                 # Global SCSS styles
│   ├── types/                  # TypeScript type definitions
│   └── App.tsx                 # Root component
├── packages/                   # Shared UI packages
│   ├── config/                 # Configuration utilities
│   └── ui/                     # Shared UI components
├── mocks/                      # MSW mock data
├── tests/                      # Test files
├── public/                     # Static assets
├── package.json                # Dependencies
├── rsbuild.config.ts           # Build configuration
├── tsconfig.json               # TypeScript config
└── vitest.config.ts            # Test configuration
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | React 19 |
| Language | TypeScript 5.8+ |
| Build Tool | Rsbuild |
| Routing | React Router v6 |
| State Management | TanStack Query (React Query) v5 |
| UI Components | react-aria-components |
| Styling | SCSS Modules |
| Animation | Motion (Framer Motion) |
| API Client | openapi-fetch |
| Testing | Vitest, Playwright, Testing Library |

## Key Features

### Type-Safe API Client

The frontend uses auto-generated TypeScript types from the OpenAPI specification:

```typescript
// Generated types ensure type safety
import createClient from 'openapi-fetch';
import type { paths } from './openapi-spec';

const client = createClient<paths>({ baseUrl: '/api' });

// Fully typed API calls
const { data, error } = await client.GET('/projects/{project_id}', {
  params: { path: { project_id: 'uuid' } }
});
```

### React Query Integration

TanStack Query provides caching, background updates, and error handling:

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Fetch projects with caching
const { data: projects, isLoading } = useQuery({
  queryKey: ['projects'],
  queryFn: () => client.GET('/projects').then(r => r.data)
});

// Mutate with optimistic updates
const queryClient = useQueryClient();
const createProject = useMutation({
  mutationFn: (name: string) => 
    client.POST('/projects', { body: { name } }),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['projects'] });
  }
});
```

### Component Architecture

Components follow a feature-based organization:

```
components/
├── common/                 # Shared components
│   ├── Button/
│   ├── Modal/
│   ├── Card/
│   └── Loading/
├── project/                # Project-related components
│   ├── ProjectCard/
│   ├── ProjectList/
│   └── ProjectForm/
├── model/                  # Model-related components
│   ├── ModelCard/
│   ├── TrainingProgress/
│   └── ModelMetrics/
├── pipeline/               # Pipeline components
│   ├── PipelineConfig/
│   ├── VideoPlayer/
│   └── AnomalyOverlay/
└── media/                  # Media management
    ├── MediaGrid/
    ├── ImageUploader/
    └── MediaPreview/
```

## Development

### Starting the Development Server

```bash
cd ui
npm start
```

The development server starts at [http://localhost:3000](http://localhost:3000) with hot module replacement enabled.

### Running with Backend

To run both frontend and backend together:

```bash
npm run dev
```

This uses `concurrently` to start both services.

### API Type Generation

When the backend API changes, regenerate the TypeScript types:

```bash
# Ensure backend is running on port 8000
npm run build:api
```

This downloads the OpenAPI spec and generates TypeScript types.

## State Management

### Query Keys

Consistent query key patterns for cache management:

```typescript
export const queryKeys = {
  projects: ['projects'] as const,
  project: (id: string) => ['projects', id] as const,
  models: (projectId: string) => ['projects', projectId, 'models'] as const,
  model: (projectId: string, modelId: string) => 
    ['projects', projectId, 'models', modelId] as const,
  media: (projectId: string) => ['projects', projectId, 'media'] as const,
  pipelines: (projectId: string) => ['projects', projectId, 'pipelines'] as const,
};
```

### Custom Hooks

Encapsulate data fetching logic in custom hooks:

```typescript
// hooks/useProjects.ts
export function useProjects() {
  return useQuery({
    queryKey: queryKeys.projects,
    queryFn: async () => {
      const { data, error } = await client.GET('/projects');
      if (error) throw error;
      return data;
    }
  });
}

// hooks/useCreateProject.ts
export function useCreateProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (name: string) => {
      const { data, error } = await client.POST('/projects', {
        body: { name }
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
    }
  });
}
```

## Routing

React Router v6 handles navigation:

```typescript
// App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/projects" element={<ProjectList />} />
        <Route path="/projects/:projectId" element={<ProjectDetail />}>
          <Route index element={<ProjectOverview />} />
          <Route path="media" element={<MediaLibrary />} />
          <Route path="models" element={<ModelList />} />
          <Route path="models/:modelId" element={<ModelDetail />} />
          <Route path="pipelines" element={<PipelineList />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

### URL State Management

Use `nuqs` for URL-based state:

```typescript
import { useQueryState } from 'nuqs';

function MediaLibrary() {
  const [filter, setFilter] = useQueryState('filter', { defaultValue: 'all' });
  const [page, setPage] = useQueryState('page', { defaultValue: '1' });
  
  // URL: /projects/123/media?filter=normal&page=2
}
```

## Styling

### SCSS Modules

Component styles are scoped using SCSS modules:

```scss
// ProjectCard.module.scss
.card {
  padding: var(--spacing-md);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  
  &:hover {
    box-shadow: var(--shadow-lg);
  }
}

.title {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
}
```

```typescript
// ProjectCard.tsx
import styles from './ProjectCard.module.scss';

function ProjectCard({ project }) {
  return (
    <div className={styles.card}>
      <h3 className={styles.title}>{project.name}</h3>
    </div>
  );
}
```

### CSS Variables

Global CSS variables ensure consistency:

```scss
// styles/variables.scss
:root {
  // Colors
  --color-primary: #0068b5;
  --color-surface: #ffffff;
  --color-background: #f5f5f5;
  --color-text: #1a1a1a;
  --color-text-muted: #6b7280;
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  
  // Spacing
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  
  // Typography
  --font-size-sm: 0.875rem;
  --font-size-md: 1rem;
  --font-size-lg: 1.25rem;
  
  // Shadows
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
}
```

## Animation

Motion (Framer Motion) provides smooth animations:

```typescript
import { motion, AnimatePresence } from 'motion/react';

function ProjectList({ projects }) {
  return (
    <AnimatePresence>
      {projects.map((project, index) => (
        <motion.div
          key={project.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ delay: index * 0.05 }}
        >
          <ProjectCard project={project} />
        </motion.div>
      ))}
    </AnimatePresence>
  );
}
```

## WebRTC Video Streaming

The frontend uses WebRTC for real-time video streaming:

```typescript
// hooks/useWebRTC.ts
export function useWebRTC(pipelineId: string) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const peerConnection = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    async function connect() {
      const pc = new RTCPeerConnection();
      peerConnection.current = pc;
      
      pc.ontrack = (event) => {
        setStream(event.streams[0]);
      };
      
      // Create and send offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      
      const { data } = await client.POST('/webrtc/offer', {
        body: { 
          pipeline_id: pipelineId,
          sdp: offer.sdp 
        }
      });
      
      await pc.setRemoteDescription(data.answer);
    }
    
    connect();
    
    return () => {
      peerConnection.current?.close();
    };
  }, [pipelineId]);

  return stream;
}
```

## Testing

### Unit Tests

Run unit tests with Vitest:

```bash
npm run test:unit

# Watch mode
npm run test:unit:watch
```

Example unit test:

```typescript
// ProjectCard.test.tsx
import { render, screen } from '@testing-library/react';
import { ProjectCard } from './ProjectCard';

describe('ProjectCard', () => {
  it('displays project name', () => {
    const project = { id: '1', name: 'Test Project' };
    
    render(<ProjectCard project={project} />);
    
    expect(screen.getByText('Test Project')).toBeInTheDocument();
  });
});
```

### Component Tests

Run Playwright component tests:

```bash
npm run test:component
```

Example component test:

```typescript
// tests/main.spec.ts
import { test, expect } from '@playwright/test';

test('creates new project', async ({ page }) => {
  await page.goto('/');
  
  await page.click('text=New Project');
  await page.fill('input[name="name"]', 'My Project');
  await page.click('text=Create');
  
  await expect(page.locator('text=My Project')).toBeVisible();
});
```

### Mocking

MSW (Mock Service Worker) provides API mocking:

```typescript
// mocks/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/projects', () => {
    return HttpResponse.json([
      { id: '1', name: 'Project 1' },
      { id: '2', name: 'Project 2' },
    ]);
  }),
];
```

## Code Quality

### Linting

ESLint enforces code style:

```bash
# Run linter
npm run lint

# Auto-fix issues
npm run lint:fix
```

### Type Checking

TypeScript ensures type safety:

```bash
npm run type-check
```

### Formatting

Prettier formats code consistently:

```bash
# Format all files
npm run format

# Check formatting
npm run format:check
```

### Import Sorting

Imports are automatically sorted by the Prettier plugin:

```typescript
// Sorted imports
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

import { Button } from '@geti/ui';

import { useProjects } from '~/hooks/useProjects';
import { ProjectCard } from '~/components/project/ProjectCard';

import styles from './Dashboard.module.scss';
```

## Build

### Production Build

```bash
npm run build
```

The production build is output to `dist/`:

```
dist/
├── index.html
├── static/
│   ├── js/
│   │   └── main.[hash].js
│   └── css/
│       └── main.[hash].css
└── assets/
```

### Preview Build

Preview the production build locally:

```bash
npm run preview
```

## Shared Packages

The frontend uses shared packages from the Geti UI:

### @geti/config

Configuration utilities and constants:

```typescript
import { API_BASE_URL, SUPPORTED_IMAGE_TYPES } from '@geti/config';
```

### @geti/ui

Shared UI components:

```typescript
import { 
  Button, 
  Modal, 
  Card, 
  Input,
  Select,
  Tooltip 
} from '@geti/ui';
```

## Best Practices

### Error Boundaries

Wrap components with error boundaries:

```typescript
import { ErrorBoundary } from 'react-error-boundary';

function App() {
  return (
    <ErrorBoundary fallback={<ErrorPage />}>
      <Routes>
        {/* ... */}
      </Routes>
    </ErrorBoundary>
  );
}
```

### Loading States

Use `spin-delay` to avoid flash of loading state:

```typescript
import { useSpinDelay } from 'spin-delay';

function ProjectList() {
  const { data, isLoading } = useProjects();
  const showSpinner = useSpinDelay(isLoading, { delay: 500, minDuration: 200 });
  
  if (showSpinner) return <LoadingSpinner />;
  return <List items={data} />;
}
```

### Accessibility

Use react-aria-components for accessible UI:

```typescript
import { Button, Dialog, Heading, Modal } from 'react-aria-components';

function ConfirmDialog({ onConfirm }) {
  return (
    <Modal>
      <Dialog>
        <Heading>Confirm Action</Heading>
        <p>Are you sure you want to proceed?</p>
        <Button onPress={onConfirm}>Confirm</Button>
      </Dialog>
    </Modal>
  );
}
```

