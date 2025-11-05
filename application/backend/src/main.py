# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import http
import os
from collections import defaultdict
from collections.abc import Sequence

import pydantic
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.responses import JSONResponse, Response

from api.endpoints.devices_endpoints import device_router
from api.endpoints.job_endpoints import job_router
from api.endpoints.media_endpoints import media_router
from api.endpoints.model_endpoints import model_router
from api.endpoints.pipeline_endpoints import router as pipeline_router
from api.endpoints.project_endpoints import project_router
from api.endpoints.sink_endpoints import router as sink_router
from api.endpoints.source_endpoints import router as source_router
from api.endpoints.trainable_models_endpoints import router as trainable_model_router
from api.endpoints.webrtc import router as webrtc_router
from core.lifecycle import lifespan
from exceptions import GetiBaseException
from services import ResourceNotFoundError
from settings import get_settings

app = FastAPI(
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    redoc_url=None,
    docs_url=None,
)

# TODO: check if middleware is required
# Enable CORS for local test UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(project_router)
app.include_router(job_router)
app.include_router(media_router)
app.include_router(model_router)
app.include_router(pipeline_router)
app.include_router(source_router)
app.include_router(sink_router)
app.include_router(webrtc_router)
app.include_router(trainable_model_router)
app.include_router(device_router)


@app.exception_handler(GetiBaseException)
def handle_base_exception(request: Request, e: GetiBaseException) -> Response:
    """
    Base exception handler
    """
    response = jsonable_encoder({"error_code": e.error_code, "message": e.message, "http_status": e.http_status})
    headers: dict[str, str] | None = None
    # 204 skipped as No Content needs to be revalidated
    if e.http_status not in [200, 201, 202, 203, 205, 206, 207, 208, 226] and request.method == "GET":
        headers = {"Cache-Control": "no-cache"}  # always revalidate
    if e.http_status in [204, 304] or e.http_status < 200:
        return Response(status_code=int(e.http_status), headers=headers)
    return JSONResponse(content=response, status_code=int(e.http_status), headers=headers)


@app.exception_handler(500)
async def handle_error(request, exception) -> JSONResponse:  # noqa: ANN001, ARG001
    """
    Handler for internal server errors
    """
    logger.error(f"Internal server error: {exception}")
    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        {"internal_server_error": "An internal server error occurred."},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers=headers,
    )


@app.exception_handler(ResourceNotFoundError)
async def handle_resource_not_found(request: Request, exception: ResourceNotFoundError) -> JSONResponse:  # noqa: ARG001
    """Handler for resource not found errors"""
    logger.error(str(exception))
    return JSONResponse(
        {"detail": exception.message},
        status_code=status.HTTP_404_NOT_FOUND,
        headers={"Cache-Control": "no-cache"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ARG001
    """
    Converts a RequestValidationError to a better readable Bad request exception.
    """
    reformatted_message = defaultdict(list)
    for pydantic_error in exc.errors():
        # `loc` usually is a list with 2 items describing the location of the error.
        # The first item specifies if the error is a body, query or path parameter and
        # the second is the parameter name. Here, only the parameter name is used along
        # with a message explaining what the problem with the parameter is.
        loc, msg = pydantic_error["loc"], pydantic_error["msg"]
        filtered_loc = loc[1:] if loc[0] in ("body", "query", "path") else loc
        field_string = ".".join(str(filtered_loc))  # nested fields with dot-notation
        reformatted_message[field_string].append(msg)

    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {
                "error_code": "bad_request",
                "message": reformatted_message,
                "http_status": http.HTTPStatus.BAD_REQUEST.value,
            }
        ),
        headers=headers,
    )


@app.exception_handler(pydantic.ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: pydantic.ValidationError) -> JSONResponse:  # noqa: ARG001
    """
    Converts a pydantic ValidationError to a better readable Bad request exception.
    """

    def format_location(loc: Sequence[str | int]) -> str:
        """
        Format location path with proper dot notation and array indices.

        Example:
            format_location(['a', 0, 'b', 1, 'c']) -> 'a[0].b[1].c'
        """
        result = ""
        for i, item in enumerate(loc):
            if isinstance(item, int):
                result += f"[{item}]"
            else:
                result += f".{item}" if i > 0 else str(item)
        return result

    errors = [
        {"message": error["msg"], "type": error["type"], "location": format_location(error.get("loc", []))}
        for error in exc.errors()
    ]

    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {
                "error_code": "invalid_payload",
                "errors": errors,
                "http_status": http.HTTPStatus.BAD_REQUEST.value,
            }
        ),
        headers=headers,
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", settings.port))
    uvicorn.run("main:app", loop="uvloop", host=settings.host, port=uvicorn_port, log_config=None)
