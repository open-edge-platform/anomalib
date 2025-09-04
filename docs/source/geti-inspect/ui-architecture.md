# Geti Edge UI Application Architecture

The Geti Edge applications aim to provide a user experience and design language consistent with the main Geti application. To achieve this, we reuse many architectural decisions from Geti.

## Goals

- **Developer Experience**: Setting up a developer environment for both the UI and server should take only seconds.
- **API Adaptability**: Adapting the UI to REST API changes should require minimal effort.
- **Consistency**: Maintain a unified look, feel, and user experience across the whole Geti ecosystem through shared design language, architectural patterns, and reusable components.

---

## Folder Structure

The application follows a modular, feature-oriented structure:

```text
.
├── packages/
│   ├── config           # Shared configuration (`@geti/config`)
│   └── ui               # Shared UI library (`@geti/ui`)
├── src/
│   ├── api/             # OpenAPI client and query hooks
│   ├── assets/          # Images, illustrations, icons
│   ├── components/      # Reusable UI components
│   ├── features/        # Application-specific feature modules
│   ├── providers.tsx    # Global providers (QueryClient, Theme, Router, etc.)
│   ├── router.tsx       # Application entrypoint (routing setup)
│   ├── routes/          # Route elements and loaders
│   └── shared/
│       ├── hooks/       # Common hooks not necessarily related to a feature
│       └── utils/       # Common utility functions not necessarily related to a feature
├── src-tauri/           # Tauri configuration
└── tests/               # Component and E2E tests
```

### Details

- **api/**: OpenAPI client and tanstack/query hooks.
- **components/**: Locally reusable components; promote mature components to `@geti/ui` where component files use `.component.tsx` suffix.
- **features/**: Feature-based modules encapsulating domain logic.
- **routes/** & **router.tsx**: Routing setup; keep route files minimal—extract complex routes into feature modules.
- **providers.tsx**: Global application providers, e.g., `QueryClientProvider`, `ThemeProvider`, `RouterProvider`.

> **Note:**
> Currently, `@geti/ui` and `@geti/config` are installed via [Degit](https://github.com/Rich-Harris/degit) to avoid npm publishing. These will be published to npm as the Geti Edge ecosystem matures.

---

## Core Pillars

- **Build system**: We use modern web tooling for React based applications
- **Application architecture**: Adapting the UI to REST API changes should require minimal effort.
- **Testing & CI/CD**: A robust testing setup that works locally and in the CI
- **AI Algorithms**: Interactive AI by implementing low latency algorithms

### Build System

Geti Edge UI uses modern web tooling for efficient development and deployment:

- **React**: Component-based UI architecture.
- **TypeScript**: Static typing for reliability and maintainability.
- **Rsbuild**: Fast and robust build toolchain for bundling, optimization, and environment targeting.
- **Tauri**: Cross-platform desktop app packaging.
- **ESLint & Prettier**: Enforced via `@geti/config` for code consistency and best practices.

---

### Application Architecture

The application leverages several libraries and architectural patterns:

- **React Router**: Single-page application navigation with dynamic routing.
- **Tanstack Query** & [`openapi-react-query`](https://openapi-ts.dev/openapi-react-query/): Server state management and type-safe API consumption.
- **@geti/ui**: Shared visual components for consistent UX.
- **React Context & Local State**: Local state via `useState`, shared state via `createContext` (for non-server state).

---

### Testing & CI/CD

Robust testing and CI/CD practices ensure reliability:

- **Vitest**: Fast unit and integration testing.
- **Playwright**: Component and end-to-end testing with shared MSW configuration for auto-mocking REST endpoints.
- **Testing Library**: User-centric React component testing; follow [guiding principles](https://testing-library.com/docs/guiding-principles) for accessibility.
- **MSW + OpenAPI**: Mock Service Worker with OpenAPI specs to simulate API responses in tests.
- **GitHub Actions**: Automated CI/CD pipelines using GitHub Actions to build, test, and deploy the application, ensuring code quality and rapid feedback.

---

### Algorithms & AI

Advanced algorithms and AI technologies enhance application capabilities:

- **@geti/smart-tools**: Suite of intelligent tools for advanced functionality and optimization.
- **WebRTC API**: Live video feeds with prediction overlays.
- **WebAssembly**: High-performance, browser-executed code for compute-intensive tasks.
- **OpenCV**: Image processing and computer vision.
- **ONNXRuntime**: In-browser machine learning model execution for predictive analytics and decision support.
