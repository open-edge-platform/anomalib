# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import engine
from db.schema import Base
from pydantic_models import Job, JobStatus, JobType, Project
from repositories import JobRepository, ProjectRepository
from services import JobService, job_service


@pytest_asyncio.fixture
async def fxt_test_db(monkeypatch):
    """Create an in-memory test database."""
    # Create in-memory SQLite database
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    test_session = async_sessionmaker(test_engine, expire_on_commit=False)

    # Initialize schema
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Patch get_async_db_session_ctx to use test database
    # Need to patch it where it's imported in JobService
    @asynccontextmanager
    async def test_get_async_db_session_ctx():
        async with test_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    # Patch both the module-level import and the engine module
    monkeypatch.setattr(job_service, "get_async_db_session_ctx", test_get_async_db_session_ctx)
    monkeypatch.setattr(engine, "get_async_db_session_ctx", test_get_async_db_session_ctx)

    try:
        yield test_engine
    finally:
        # Cleanup
        await test_engine.dispose()


@pytest_asyncio.fixture
async def fxt_test_project(fxt_test_db):
    """Create a test project in the database."""
    project = Project(name="Test Project")
    async_session = async_sessionmaker(fxt_test_db, expire_on_commit=False)
    async with async_session() as session:
        repo = ProjectRepository(session)
        saved_project = await repo.save(project)
        await session.commit()
        return saved_project


@pytest_asyncio.fixture
async def fxt_jobs_in_db(fxt_test_db, fxt_test_project):
    """Create multiple jobs in the test database."""
    jobs = []
    async_session = async_sessionmaker(fxt_test_db, expire_on_commit=False)
    async with async_session() as session:
        repo = JobRepository(session)
        # Create 15 jobs with different statuses and types
        for i in range(15):
            job = Job(
                project_id=fxt_test_project.id,
                type=JobType.TRAINING if i % 2 == 0 else JobType.OPTIMIZATION,
                status=JobStatus.PENDING if i < 5 else (JobStatus.RUNNING if i < 10 else JobStatus.COMPLETED),
                payload={"model_name": f"model_{i}", "index": i},
                message=f"Test job {i}",
            )
            saved_job = await repo.save(job)
            jobs.append(saved_job)
        await session.commit()
    return jobs


class TestJobServicePagination:
    @pytest.mark.parametrize(
        "limit,offset,expected_count,expected_start_index",
        [
            (5, 0, 5, 0),  # First page
            (5, 5, 5, 5),  # Second page
            (5, 10, 5, 10),  # Third page
            (10, 0, 10, 0),  # Larger page
            (10, 10, 5, 10),  # Second large page (partial)
            (20, 0, 15, 0),  # Page larger than total
            (5, 15, 0, None),  # Offset beyond total (empty result)
            (3, 12, 3, 12),  # Partial last page
            (15, 0, 15, 0),  # Exact total
        ],
    )
    @pytest.mark.asyncio
    async def test_get_job_list_pagination(
        self,
        fxt_test_db,
        fxt_jobs_in_db,
        limit,
        offset,
        expected_count,
        expected_start_index,
    ):
        """Test pagination with various limit/offset combinations."""
        result = await JobService.get_job_list(limit=limit, offset=offset)

        # Verify pagination metadata
        assert result.pagination.limit == limit
        assert result.pagination.offset == offset
        assert result.pagination.count == expected_count
        assert result.pagination.total == len(fxt_jobs_in_db)

        # Verify number of items returned
        assert len(result.jobs) == expected_count

        # Verify items are correct subset (if not empty)
        if expected_count > 0 and expected_start_index is not None:
            # Jobs should be returned in order (by creation time, which is insertion order)
            # Verify the first job matches expected index
            expected_job = fxt_jobs_in_db[expected_start_index]
            assert result.jobs[0].id == expected_job.id

            # Verify the last job matches expected index
            if expected_count > 1:
                expected_last_index = min(expected_start_index + expected_count - 1, len(fxt_jobs_in_db) - 1)
                expected_last_job = fxt_jobs_in_db[expected_last_index]
                assert result.jobs[-1].id == expected_last_job.id

    @pytest.mark.asyncio
    async def test_get_job_list_empty_database(self, fxt_test_db, fxt_test_project):
        """Test pagination with empty database."""
        result = await JobService.get_job_list(limit=10, offset=0)

        assert result.pagination.limit == 10
        assert result.pagination.offset == 0
        assert result.pagination.count == 0
        assert result.pagination.total == 0
        assert len(result.jobs) == 0

    @pytest.mark.asyncio
    async def test_get_job_list_offset_beyond_total(self, fxt_test_db, fxt_jobs_in_db):
        """Test pagination when offset is beyond total number of items."""
        total_jobs = len(fxt_jobs_in_db)
        result = await JobService.get_job_list(limit=10, offset=total_jobs + 10)

        assert result.pagination.limit == 10
        assert result.pagination.offset == total_jobs + 10
        assert result.pagination.count == 0
        assert result.pagination.total == total_jobs
        assert len(result.jobs) == 0

    @pytest.mark.asyncio
    async def test_get_job_list_exact_total(self, fxt_test_db, fxt_jobs_in_db):
        """Test pagination when limit equals total number of items."""
        total_jobs = len(fxt_jobs_in_db)
        result = await JobService.get_job_list(limit=total_jobs, offset=0)

        assert result.pagination.limit == total_jobs
        assert result.pagination.offset == 0
        assert result.pagination.count == total_jobs
        assert result.pagination.total == total_jobs
        assert len(result.jobs) == total_jobs

    @pytest.mark.asyncio
    async def test_get_job_list_pagination_ordering(self, fxt_test_db, fxt_jobs_in_db):
        """Test that pagination returns jobs in consistent order."""
        # Get first page
        page1 = await JobService.get_job_list(limit=5, offset=0)
        # Get second page
        page2 = await JobService.get_job_list(limit=5, offset=5)

        # Verify no overlap
        page1_ids = {job.id for job in page1.jobs}
        page2_ids = {job.id for job in page2.jobs}
        assert page1_ids.isdisjoint(page2_ids)

        # Verify all jobs from both pages are in original list
        all_page_ids = page1_ids | page2_ids
        all_job_ids = {job.id for job in fxt_jobs_in_db}
        assert all_page_ids.issubset(all_job_ids)
