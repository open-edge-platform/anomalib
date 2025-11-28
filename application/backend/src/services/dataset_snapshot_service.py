# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import os
import tempfile
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anomalib.data.utils import LabelName
from loguru import logger
from PIL import Image
from sqlalchemy import func, select

from db import get_async_db_session_ctx
from db.schema import ModelDB
from pydantic_models import DatasetSnapshot
from repositories import DatasetSnapshotRepository, MediaRepository, ProjectRepository
from repositories.binary_repo import DatasetSnapshotBinaryRepository, ImageBinaryRepository


class DatasetSnapshotService:
    @classmethod
    async def create_snapshot(cls, project_id: UUID) -> DatasetSnapshot:
        """Create a new immutable dataset snapshot from current media."""
        snapshot_id = uuid4()
        filename = f"{snapshot_id}.parquet"

        logger.info(f"Creating dataset snapshot {snapshot_id} for project {project_id}")

        async with get_async_db_session_ctx() as session:
            media_repo = MediaRepository(session, project_id=project_id)

            image_bin_repo = ImageBinaryRepository(project_id=project_id)

            data_rows = []
            for media in await media_repo.get_all():
                # Read bytes
                try:
                    img_bytes = await image_bin_repo.read_file(media.filename)
                except FileNotFoundError:
                    logger.error(f"Image file {media.filename} missing for media {media.id}, skipping")
                    continue

                label_index = LabelName.ABNORMAL if media.is_anomalous else LabelName.NORMAL

                data_rows.append({
                    "image": img_bytes,
                    "label_index": label_index,
                    "original_image_path": media.filename,  # Virtual path, useful for debugging
                    # "mask": ... # TODO: Add mask support if we have masks
                })

            # Create Table
            if not data_rows:
                logger.warning(f"Creating snapshot for project {project_id} with no media")
                # Create empty table with schema
                schema = pa.schema([
                    ("image", pa.binary()),
                    ("label_index", pa.int64()),
                    ("original_image_path", pa.string()),
                ])
                table = pa.Table.from_pydict({}, schema=schema)
            else:
                # Convert to PyArrow Table
                pydict = {
                    "image": [r["image"] for r in data_rows],
                    "label_index": [r["label_index"] for r in data_rows],
                    "original_image_path": [r["original_image_path"] for r in data_rows],
                }
                table = pa.Table.from_pydict(pydict)

        # Write Parquet file
        snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)

        def _write_parquet() -> bytes:
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            return sink.getvalue().to_pybytes()

        parquet_bytes = await asyncio.to_thread(_write_parquet)

        try:
            await snapshot_bin_repo.save_file(filename, parquet_bytes)
            logger.info(f"Saved snapshot file {filename}")

            # Create snapshot DB Record
            async with get_async_db_session_ctx() as session:
                snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
                snapshot = DatasetSnapshot(
                    id=snapshot_id,
                    project_id=project_id,
                    filename=filename,
                )
                return await snapshot_repo.save(snapshot)

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            # Cleanup file if saved
            try:
                await snapshot_bin_repo.delete_file(filename)
                logger.info(f"Deleted incomplete snapshot file {filename}")
            except FileNotFoundError:
                logger.info(f"No incomplete snapshot file {filename} to delete")
            raise e

    @classmethod
    async def get_or_create_snapshot(cls, project_id: UUID, snapshot_id: UUID | None) -> DatasetSnapshot:
        """Get existing snapshot or create a new one:
        - If snapshot_id is provided, fetch and return it.
        - If snapshot_id is None, create a new snapshot if necessary.
        - If dataset has not being updated since last snapshot, return existing snapshot.
        """
        async with get_async_db_session_ctx() as session:
            snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
            if snapshot_id is not None:
                snapshot = await snapshot_repo.get_by_id(snapshot_id)
                if snapshot is None:
                    raise ValueError(f"Snapshot {snapshot_id} not found in project {project_id}")
                return snapshot

            dataset_updated_at = await ProjectRepository(session).get_dataset_timestamp(project_id=project_id)
            latest_snapshot = await snapshot_repo.get_latest_snapshot()
            if latest_snapshot and latest_snapshot.created_at >= dataset_updated_at:
                logger.info(f"Using existing up-to-date snapshot `{latest_snapshot.id}` in project `{project_id}`")
                return latest_snapshot
        # Create new snapshot only if no up-to-date snapshot exists
        logger.info(f"Creating new snapshot for project `{project_id}`")
        return await cls.create_snapshot(project_id=project_id)

    @classmethod
    async def delete_snapshot_if_unused(cls, snapshot_id: UUID, project_id: UUID) -> None:
        """Delete snapshot only if no models reference it."""
        async with get_async_db_session_ctx() as session:
            # Check references
            stmt = select(func.count(ModelDB.id)).where(ModelDB.dataset_snapshot_id == str(snapshot_id))
            count = (await session.execute(stmt)).scalar() or 0

            if count > 0:
                logger.info(f"Snapshot {snapshot_id} is used by {count} models. Skipping delete.")
                return

            logger.info(f"Snapshot {snapshot_id} is unused. Deleting.")

            # Delete DB Record
            snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
            snapshot = await snapshot_repo.get_by_id(snapshot_id)

            if snapshot:
                await snapshot_repo.delete_by_id(snapshot_id)

        # Delete file (outside DB session)
        if snapshot:
            snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
            try:
                await snapshot_bin_repo.delete_file(snapshot.filename)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Error deleting snapshot file {snapshot.filename}: {e}")

    @staticmethod
    def extract_snapshot_to_path(snapshot_path: str, temp_dir: str) -> None:
        """Extract images from Parquet snapshot to a temporary directory structure."""
        logger.info(f"Extracting snapshot {snapshot_path} to {temp_dir}")

        # Ensure directories exist
        normal_dir = os.path.join(temp_dir, "normal")
        abnormal_dir = os.path.join(temp_dir, "abnormal")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)

        # Read parquet
        df = pd.read_parquet(snapshot_path)

        for idx, row in enumerate(df.itertuples(index=False)):
            try:
                # Determine label
                is_anomalous = False
                if hasattr(row, "label_index"):
                    is_anomalous = row.label_index == 1
                elif hasattr(row, "label"):
                    is_anomalous = str(row.label).lower() in {"abnormal", "anomalous"}

                target_dir = abnormal_dir if is_anomalous else normal_dir

                # Determine filename
                filename = f"image_{idx}.png"
                if original_path := getattr(row, "original_image_path", None):
                    filename = os.path.basename(str(original_path))

                # Handle duplicates
                save_path = os.path.join(target_dir, filename)
                if os.path.exists(save_path):
                    base, ext = os.path.splitext(filename)
                    save_path = os.path.join(target_dir, f"{base}_{idx}{ext}")

                # Write image
                image_data = getattr(row, "image", None)
                if image_data and isinstance(image_data, (bytes, bytearray)):
                    image = Image.open(io.BytesIO(image_data))
                    image.save(save_path)

            except Exception as e:
                logger.warning(f"Failed to extract image at index {idx}: {e}")

    @classmethod
    @asynccontextmanager
    async def use_snapshot_as_folder(cls, snapshot_id: UUID, project_id: UUID):
        """Context manager to use a snapshot as a temporary folder."""
        snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
        snapshot_path = snapshot_bin_repo.get_snapshot_path(snapshot_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            await asyncio.to_thread(cls.extract_snapshot_to_path, snapshot_path, temp_dir)
            yield temp_dir
