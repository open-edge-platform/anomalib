# Getting Started

This guide will help you set up the Geti Inspect application for development and get it running on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Installation |
|-------------|---------|--------------|
| Python | 3.13+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 24.2.0+ | [nodejs.org](https://nodejs.org/) |
| npm | 11.3.0+ | Included with Node.js |
| uv | Latest | [astral.sh/uv](https://docs.astral.sh/uv/) |

### Installing uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib/application
```

### 2. Backend Setup

Navigate to the backend directory and install dependencies:

```bash
cd backend

# Create virtual environment and install dependencies
uv sync

# Initialize the database
uv run src/cli.py init-db
```

### 3. Frontend Setup

In a new terminal, navigate to the UI directory:

```bash
cd application/ui

# Install dependencies (this will also clone required Geti UI packages)
npm install
```

## Running the Application

### Option 1: Run Both Services Together

From the `ui` directory:

```bash
npm run dev
```

This starts both the backend server and frontend development server concurrently.

### Option 2: Run Services Separately

**Terminal 1 - Backend:**
```bash
cd backend
./run.sh
```

The backend server starts at [http://localhost:8000](http://localhost:8000).

**Terminal 2 - Frontend:**
```bash
cd ui
npm start
```

The frontend development server starts at [http://localhost:3000](http://localhost:3000).

## Verifying the Installation

1. Open [http://localhost:3000](http://localhost:3000) in your browser
2. You should see the Geti Inspect dashboard
3. The API documentation is available at [http://localhost:8000/api/openapi.json](http://localhost:8000/api/openapi.json)

## Your First Project

### 1. Create a New Project

1. Click **"New Project"** in the dashboard
2. Enter a project name and description
3. Click **"Create"**

### 2. Add Training Images

1. Navigate to your project's **Media** tab
2. Click **"Upload Images"**
3. Select normal (non-anomalous) images for training
4. Optionally add anomalous images for validation

### 3. Train a Model

1. Go to the **Models** tab
2. Click **"Train New Model"**
3. Select an algorithm (e.g., PatchCore, EfficientAD)
4. Configure training parameters
5. Click **"Start Training"**

### 4. Run Inference

1. Once training completes, go to the **Pipelines** tab
2. Create a new pipeline with your trained model
3. Select an input source (webcam, video file, or IP camera)
4. Start the pipeline to see real-time anomaly detection

## Development Workflow

### Backend Development

```bash
cd backend

# Run the server with auto-reload
uv run src/main.py

# Run tests
uv run pytest

# Run linting
# Run all linting checks via pre-commit (recommended)
uv run pre-commit run --all-files

# Optionally, you can run individual tools directly:
uv run ruff check src/
uv run mypy src/
```

### Frontend Development

```bash
cd ui

# Start development server
npm start

# Run tests
npm run test:unit

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format
```

### Database Management

```bash
cd backend

# Check database status
uv run src/cli.py check-db

# Run migrations
uv run src/cli.py migrate

# Create a new migration
uv run src/cli.py create-db-revision -m "description of changes"

# Seed with test data
uv run src/cli.py seed

# Clean the database
uv run src/cli.py clean-db
```

### API Development

After making changes to API endpoints, regenerate the OpenAPI types for the frontend:

```bash
# From the backend, start the server first
./run.sh

# In the ui directory
npm run build:api
```

## Project Structure

```
application/
├── backend/
│   ├── src/
│   │   ├── api/           # REST API endpoints
│   │   ├── core/          # Application lifecycle and config
│   │   ├── db/            # Database models and migrations
│   │   ├── entities/      # Domain entities
│   │   ├── pydantic_models/  # Request/Response schemas
│   │   ├── repositories/  # Data access layer
│   │   ├── services/      # Business logic
│   │   ├── utils/         # Utility functions
│   │   ├── webrtc/        # WebRTC streaming
│   │   └── workers/       # Background workers
│   ├── tests/             # Backend tests
│   ├── pyproject.toml     # Python dependencies
│   └── run.sh             # Startup script
│
├── ui/
│   ├── src/
│   │   ├── api/           # API client and types
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── pages/         # Page components
│   │   └── styles/        # Global styles
│   ├── packages/          # Shared UI packages
│   ├── tests/             # Frontend tests
│   └── package.json       # Node.js dependencies
│
└── docs/                  # Documentation (you are here)
```

## Troubleshooting

### Common Issues

**Backend fails to start:**
- Ensure Python 3.13+ is installed: `python --version`
- Check if uv is installed: `uv --version`
- Verify the virtual environment: `uv sync`

**Frontend fails to start:**
- Ensure Node.js 24+ is installed: `node --version`
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

**Database errors:**
- Re-initialize the database: `uv run src/cli.py init-db`
- Check database status: `uv run src/cli.py check-db`

**WebRTC streaming not working:**
- Ensure the backend is running on port 8000
- Check browser console for WebRTC errors
- Verify camera permissions are granted

### Getting Help

- Check the [anomalib GitHub issues](https://github.com/open-edge-platform/anomalib/issues)
- Review the [API Reference](./api-reference.md) for endpoint details
- See [Configuration](./configuration.md) for environment variables

## Next Steps

- Read the [Backend Documentation](./backend.md) for API development
- Read the [Frontend Documentation](./frontend.md) for UI development
- Review the [Configuration Guide](./configuration.md) for deployment options

