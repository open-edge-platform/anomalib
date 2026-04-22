# REST API Checklist (FastAPI)

Use this checklist during endpoint creation or API review.

## Resource design

- [ ] Paths use nouns (no action verbs in URL segments).
- [ ] Collection routes are plural and consistent.
- [ ] Nested resources are logical and not too deep.
- [ ] Route names are lowercase and stable.

## HTTP semantics

- [ ] HTTP method matches operation intent.
- [ ] Success status codes are correct (`200/201/204`).
- [ ] Error status codes are correct (`400/401/403/404/409/422/500`).
- [ ] `PUT` vs `PATCH` semantics are correctly applied.

## Contracts and validation

- [ ] Request models validate required constraints.
- [ ] Response models are explicit and documented.
- [ ] Error response shape is consistent.
- [ ] JSON is used consistently for API payloads.

## FastAPI implementation

- [ ] Routers are organized by domain/resource.
- [ ] Dependencies are injected with `Depends(...)`.
- [ ] Business logic is outside endpoint handlers.
- [ ] Domain exceptions are mapped to HTTP responses cleanly.
- [ ] OpenAPI `responses` metadata includes custom error cases where needed.

## Security and operability

- [ ] Authentication is enforced where required.
- [ ] Authorization checks follow least privilege.
- [ ] Sensitive internals are not leaked in errors.
- [ ] Collection endpoints support pagination/filtering/sorting when needed.
- [ ] API versioning strategy exists for breaking changes.
