# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import http
from collections import defaultdict
from collections.abc import Sequence

import pydantic
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from loguru import logger
from starlette.responses import JSONResponse, Response

from exceptions import GetiBaseException
from main import app
from services import ActivePipelineConflictError, ResourceNotFoundError


@app.exception_handler(GetiBaseException)
def handle_base_exception(request: Request, e: GetiBaseException) -> Response:
    """
    Base exception handler
    """
    response = jsonable_encoder({"error_code": e.error_code, "message": e.message, "http_status": e.http_status})
    headers: dict[str, str] | None = None
    # 204 skipped as No Content needs to be revalidated
    if e.http_status not in {200, 201, 202, 203, 205, 206, 207, 208, 226} and request.method == "GET":
        headers = {"Cache-Control": "no-cache"}  # always revalidate
    if e.http_status in {204, 304} or e.http_status < 200:
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
        filtered_loc = loc[1:] if loc[0] in {"body", "query", "path"} else loc
        field_string = ".".join(str(item) for item in filtered_loc)  # nested fields with dot-notation
        reformatted_message[field_string].append(msg)

    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({
            "error_code": "bad_request",
            "message": reformatted_message,
            "http_status": http.HTTPStatus.BAD_REQUEST.value,
        }),
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
        content=jsonable_encoder({
            "error_code": "invalid_payload",
            "errors": errors,
            "http_status": http.HTTPStatus.BAD_REQUEST.value,
        }),
        headers=headers,
    )


@app.exception_handler(ActivePipelineConflictError)
async def handle_active_pipeline_conflict(
    request: Request,  # noqa: ARG001
    exception: ActivePipelineConflictError,
) -> JSONResponse:
    """Handler for active pipeline conflict errors"""
    logger.error(exception.message)
    return JSONResponse(
        {"detail": exception.message},
        status_code=status.HTTP_409_CONFLICT,
        headers={"Cache-Control": "no-cache"},
    )
