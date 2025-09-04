# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import http
from uuid import UUID


class GetiBaseException(Exception):
    """
    Base class for Geti exceptions with a predefined HTTP error code.

    :param message: str message providing short description of error
    :param error_code: str id of error
    :param http_status: int default http status code to return to user
    """

    def __init__(self, message: str, error_code: str, http_status: int) -> None:
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(message)


class InvalidMediaException(GetiBaseException):
    """
    Exception raised when uploaded media file is invalid.

    :param message: str containing a custom message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="invalid_media",
            http_status=http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )


class PayloadTooLargeException(GetiBaseException):
    """
    Exception raised when the request payload is too large.

    :param max_size: Max size in MB
    """

    def __init__(self, max_size: float) -> None:
        super().__init__(
            message=f"Request too large: exceeding {max_size} MB is not allowed.",
            error_code="payload_too_large",
            http_status=http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class DuplicateJobException(GetiBaseException):
    """
    Exception raised when attempting to submit a duplicate job.

    :param message: str containing a custom message about the duplicate job.
    """

    def __init__(self, message: str = "A job with the same payload is already running or queued") -> None:
        super().__init__(
            message=message,
            error_code="duplicate_job",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ResourceNotFoundException(GetiBaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_id: str | UUID, resource_name: str) -> None:
        super().__init__(
            message=f"The requested {resource_name} could not be found. {resource_name.title()} ID: `{resource_id}`.",
            error_code=f"{resource_name}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )
