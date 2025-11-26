# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import abc
import asyncio
import os
import shutil
from enum import StrEnum
from functools import cached_property
from uuid import UUID

from anomalib.deploy import ExportType

STORAGE_ROOT_PATH = "data"


class FileType(StrEnum):
    IMAGES = "images"
    MODELS = "models"
    SNAPSHOTS = "snapshots"


class BinaryRepository(metaclass=abc.ABCMeta):
    def __init__(self, project_id: str | UUID, file_type: FileType):
        self.project_id = str(project_id)
        self.file_type = file_type

    async def read_file(self, filename: str) -> bytes:
        """
        Read a binary file from the filesystem.

        :param filename: Relative path to the file.
        :return: Binary content of the file.
        """

        def stdlib_read():
            full_path = self.get_full_path(filename)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")
            with open(full_path, "rb") as fp:
                return fp.read()

        return await asyncio.to_thread(stdlib_read)

    @cached_property
    def project_folder_path(self) -> str:
        """
        Get the project folder path containing the binary files.
        """
        return os.path.join(STORAGE_ROOT_PATH, self.file_type, "projects", self.project_id)

    @abc.abstractmethod
    def get_full_path(self, filename: str) -> str:
        """
        Get the full path for a given filename within the project folder.

        :param filename: Name of the file.
        :return: Full path to the file.
        """

    async def save_file(self, filename: str, content: bytes) -> str:
        """
        Save a binary file to the filesystem under the project directory.

        :param filename: Name of the file to save.
        :param content: Binary content of the file.
        :return: The path where the file was saved.
        """

        def stdlib_write():
            full_path = self.get_full_path(filename)
            folder, _ = full_path.split(filename)
            os.makedirs(folder, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(content)
            return full_path

        try:
            destination_path = await asyncio.to_thread(stdlib_write)
        except Exception as e:
            raise OSError(f"Failed to save {self.file_type} file: {filename}") from e
        return destination_path

    async def delete_file(self, filename: str) -> None:
        """
        Delete a binary file from the filesystem.

        :param filename: Name of the file to delete.
        """

        def stdlib_delete():
            full_path = self.get_full_path(filename)
            if os.path.isfile(full_path):
                os.remove(full_path)
            else:
                raise FileNotFoundError(f"File not found: {full_path}")

        await asyncio.to_thread(stdlib_delete)


class DatasetSnapshotBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.SNAPSHOTS)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)

    def get_snapshot_path(self, snapshot_id: str | UUID) -> str:
        """
        Get the full path for a dataset snapshot.

        :param snapshot_id: ID of the snapshot.
        :return: Full path to the snapshot file.
        """
        return self.get_full_path(f"{snapshot_id}.parquet")


class ImageBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.IMAGES)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)


class ModelBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID, model_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.MODELS)
        self._model_id = str(model_id)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.model_folder_path, filename)

    @cached_property
    def model_folder_path(self) -> str:
        """
        Get the folder path for models.

        :return: Folder path for models.
        """
        return os.path.join(self.project_folder_path, self._model_id)

    async def delete_model_folder(self) -> None:
        """
        Delete a model folder from the filesystem.
        """

        def stdlib_delete_folder():
            folder_path = self.model_folder_path
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            else:
                raise FileNotFoundError(f"Model folder not found: {folder_path}")

        await asyncio.to_thread(stdlib_delete_folder)

    def get_weights_file_path(self, format: ExportType, name: str) -> str:
        """
        Read a weights file from the model folder.

        :param format: Format of the model (e.g., ExportType.OPENVINO).
        :param name: Name of the weights to read.
        :return: path of the weights file.
        """
        return os.path.join(self.model_folder_path, "weights", format, name)
