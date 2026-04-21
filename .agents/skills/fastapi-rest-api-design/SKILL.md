---
name: fastapi-rest-api-design
description: Designs and reviews REST APIs for FastAPI services using consistent resource naming, HTTP semantics, validation, security, and error handling patterns. Use for backend API tasks, endpoint design/refactors, or API review requests in FastAPI/Python projects.
---

# FastAPI REST API Design

Use this skill when designing, implementing, or reviewing FastAPI REST endpoints.

## Purpose and scope

This skill enforces practical REST design and FastAPI implementation quality: resource naming, HTTP semantics, schema quality,
dependency boundaries, security checks, and concise review output.

## Core design rules

### Resource and path conventions

- Use nouns in paths, not action verbs.
- Use plural collection names such as `/projects`, `/users`.
- Use stable item identifiers such as `/projects/{project_id}`.
- Keep names lowercase and consistent.
- Keep nesting shallow (typically max 2 levels), for example `/projects/{project_id}/models`.
- Do not expose internal storage structure in public routes.

### HTTP methods and status semantics

- Use `GET` read, `POST` create, `PUT` full replace, `PATCH` partial update, `DELETE` remove.
- Return `201` on create, `200` on standard success, and `204` when no response body is needed.
- Choose precise error codes (`400`, `401`, `403`, `404`, `409`, `422`, `500`) and avoid vague fallbacks.
- Keep semantics consistent across related endpoints.

### Request/response contracts

- Use JSON payloads for standard API requests/responses.
- Use Pydantic models for request and response contracts.
- Enforce explicit field constraints (enums, lengths, ranges, formats).
- Prefer explicit response models for stable contracts.
- Keep error response bodies consistent across endpoints.

### FastAPI architecture patterns

- Organize routes by resource/domain with `APIRouter`.
- Inject dependencies with `Depends(...)`; avoid hidden globals.
- Keep handlers thin and move business logic into services/use-cases.
- Raise domain exceptions in service layer and map to HTTP errors at API boundary.
- Add explicit `responses={...}` metadata when custom errors must appear in OpenAPI docs.

### Security and operability

- Enforce HTTPS in deployed environments.
- Enforce authentication and authorization per route/use case.
- Apply least-privilege checks for each resource operation.
- Avoid exposing sensitive internals in error details.
- Add filtering, sorting, and pagination for large collection endpoints.
- Introduce versioning (for example `/v1/...`) before shipping breaking API changes.

## Review workflow

1. Classify each endpoint as collection, item, or nested resource.
2. Verify verb-to-action mapping.
3. Verify status code and error semantics.
4. Verify Pydantic schema quality and response model clarity.
5. Verify DI and layer boundaries.
6. Verify authn/authz and least-privilege behavior.
7. Apply checklist and report material issues first.

## Output style

When asked to design or review APIs, respond concisely with:

1. Endpoint proposal (route + method list).
2. Contract notes (request/response + validation).
3. Security checks (authn/authz + sensitive data handling).
4. Checklist verdict (pass/fail highlights).
5. Top fixes in priority order.

## Additional resources

- Checklist: [REST_API_CHECKLIST.md](REST_API_CHECKLIST.md)
