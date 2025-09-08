# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from fastapi import File, Request, UploadFile

from exceptions import InvalidMediaException, PayloadTooLargeException
from models import ImageExtension


class MediaRestValidator:
    SUPPORTED_IMAGE_TYPES = [extension.value for extension in ImageExtension]
    MAX_BYTES_SIZE = 4.7 * 1024**3

    @staticmethod
    def validate_image_file(request: Request, file: UploadFile = File(None)) -> UploadFile | None:
        """
        Validates a request to upload an image, if the file is not present, returns None.

        :param request: FastAPI request that was made to upload the file
        :param file: uploaded image file
        :raises InvalidMediaException if the file name is empty or the file extension is
        not in the supported file extensions
        :raises PayloadTooLargeException when the total size of the request exceeds 8GB
        """
        if file is None:
            return None

        file_size = file.size if file.size else int(request.headers["content-length"])
        if file_size > MediaRestValidator.MAX_BYTES_SIZE:
            raise PayloadTooLargeException(MediaRestValidator.MAX_BYTES_SIZE / (1024**2))

        if file.filename is None:
            raise InvalidMediaException("Filename can't be empty.")

        extension = list(os.path.splitext(file.filename)).pop().lower()
        if extension not in MediaRestValidator.SUPPORTED_IMAGE_TYPES:
            raise InvalidMediaException(
                f"Not a valid image format. Received {extension}, but only the following are allowed: "
                f"{str(MediaRestValidator.SUPPORTED_IMAGE_TYPES)}"
            )
        return file
