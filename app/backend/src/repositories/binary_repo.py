# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from enum import StrEnum
from uuid import UUID

STORAGE_ROOT_PATH = "data"
IMAGES_PATH = "images"
MODELS_PATH = "models"


class FILETYPE(StrEnum):
    IMAGES = "images"
    MODELS = "models"


class BinaryRepository:
    def __init__(self, project_id: str | UUID):
        self.project_id = str(project_id)

    @staticmethod
    async def read_file(path: str) -> bytes:
        """
        Read a binary file from the filesystem.

        :param path: Relative path to the file.
        :return: Binary content of the file.
        """

        def stdlib_read():
            full_path = os.path.join(STORAGE_ROOT_PATH, path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")
            with open(full_path, "rb") as fp:
                return fp.read()

        return await asyncio.to_thread(stdlib_read)

    def get_full_path(self, filename: str, file_type: FILETYPE) -> str:
        """
        Get the full filesystem path for a given relative path.

        :param filename: name of the file.
        :param file_type: Type of the file
        :return: Full filesystem path.
        """
        return os.path.join(STORAGE_ROOT_PATH, file_type, "projects", self.project_id, filename)

    async def save_file(self, filename: str, content: bytes, file_type: FILETYPE) -> str:
        """
        Save a binary file to the filesystem under the project directory.

        :param filename: Name of the file to save.
        :param content: Binary content of the file.
        :param file_type: Type of the file
        :return: The path where the file was saved.
        """

        def stdlib_write():
            full_path = self.get_full_path(filename=filename, file_type=file_type)
            folder, _ = full_path.split(filename)
            os.makedirs(folder, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(content)
            return full_path

        try:
            destination_path = await asyncio.to_thread(stdlib_write)
        except Exception as e:
            raise OSError(f"Failed to save {file_type.value} file: {filename}") from e
        return destination_path

    async def delete_file(self, filename: str, file_type: FILETYPE) -> None:
        """
        Delete a binary file from the filesystem.

        :param filename: Name of the file to delete.
        :param file_type: Type of the file
        """

        def stdlib_delete():
            full_path = self.get_full_path(filename=filename, file_type=file_type)
            if os.path.isfile(full_path):
                os.remove(full_path)
            else:
                raise FileNotFoundError(f"File not found: {full_path}")

        await asyncio.to_thread(stdlib_delete)
