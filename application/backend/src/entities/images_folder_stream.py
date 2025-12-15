# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from collections.abc import Callable

import cv2
from loguru import logger
from watchdog.events import DirCreatedEvent, DirDeletedEvent, FileCreatedEvent, FileDeletedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from entities.stream_data import StreamData
from entities.video_stream import VideoStream
from pydantic_models.source import SourceType


class ImagesFolderEventHandler(FileSystemEventHandler):
    """
    Images folder FS event handler, watches for files adding or removing
    """

    def __init__(self, on_file_added: Callable[[str], None], on_file_deleted: Callable[[str], None]):
        self.on_file_added = on_file_added
        self.on_file_deleted = on_file_deleted

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        if isinstance(event, FileCreatedEvent):
            path = event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
            self.on_file_added(path)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        if isinstance(event, FileDeletedEvent):
            path = event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
            self.on_file_deleted(path)


class ImagesFolderStream(VideoStream):
    """Video stream implementation using images folder."""

    def __init__(self, folder_path: str, ignore_existing_images: bool) -> None:
        """Initialize images folder stream.
        Args:
            folder_path (str): Path to the folder with images
            ignore_existing_images (bool): Flag if images, which already are in the folder at startup moment,
            should be ignored
        """
        self.folder_path = folder_path
        logger.info(f"Using folder_path: {self.folder_path}")

        self.files = []
        if not ignore_existing_images:
            self.files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]
            self.files.sort(key=os.path.getmtime)
        logger.info(f"Folder contains {len(self.files)} files")

        # Lock for thread-safe operations, required by watchdog thread(s)
        self.files_lock = threading.Lock()

        self._init_watchdog(folder_path)

    def _init_watchdog(self, folder_path: str) -> None:
        """
        Initialize the watchdog to monitor images folder events.

        Args:
            folder_path: path to the folder with images
        """
        event_handler = ImagesFolderEventHandler(
            self.file_added,
            self.file_deleted,
        )

        self.observer = Observer()
        self.observer.schedule(event_handler, folder_path, recursive=True)
        self.observer.start()

    def file_added(self, path: str) -> None:
        logger.debug(f"Added file {path}")
        if path not in self.files:
            with self.files_lock:
                self.files.append(path)

    def file_deleted(self, path: str) -> None:
        logger.debug(f"Deleted file {path}")
        if path in self.files:
            with self.files_lock:
                self.files.remove(path)

    def get_data(self) -> StreamData | None:
        try:
            file = self.files.pop(0)
            image = cv2.imread(file)
            if image is None:
                # Image cannot be loaded
                return None
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return StreamData(
                frame_data=image,
                timestamp=os.path.getmtime(file),
                source_metadata={
                    "source_type": SourceType.IMAGES_FOLDER.value,
                    "folder_path": self.folder_path,
                },
            )
        except IndexError:
            return None

    def is_real_time(self) -> bool:
        return False

    def release(self) -> None:
        self.observer.stop()
        self.observer.join()
