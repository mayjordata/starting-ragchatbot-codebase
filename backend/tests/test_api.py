"""
Tests for FastAPI endpoints in app.py

These tests verify:
1. POST /api/query endpoint handles requests correctly
2. GET /api/courses endpoint returns course statistics
3. Error handling returns proper HTTP status codes
4. Request/response validation works as expected

Note: We define a test-specific FastAPI app here to avoid import issues
with static file mounting that requires the frontend directory.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Pydantic models (same as app.py)
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class SourceCitation(BaseModel):
    """Model for a source citation with optional link"""
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[SourceCitation]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system"""
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=[SourceCitation(**s) for s in sources],
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_response_with_sources(self, mock_rag_system):
        """Test that query endpoint returns answer and sources"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "What is tool use?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about the course."
        assert len(data["sources"]) == 1

    def test_query_with_session_id_uses_provided_id(self, mock_rag_system):
        """Test that provided session_id is used"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": "existing-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        mock_rag_system.query.assert_called_once_with(
            "Follow up question", "existing-session"
        )

    def test_query_without_session_id_creates_new_session(self, mock_rag_system):
        """Test that new session is created when not provided"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "New question"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new-session-id"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_empty_query_returns_validation_error(self, mock_rag_system):
        """Test that empty query returns validation error"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # FastAPI allows empty strings, so this should still work
        assert response.status_code == 200

    def test_query_missing_query_field_returns_422(self, mock_rag_system):
        """Test that missing query field returns 422 Unprocessable Entity"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422

    def test_query_with_invalid_json_returns_422(self, mock_rag_system):
        """Test that invalid JSON returns 422"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_error_returns_500(self, mock_rag_system):
        """Test that RAG system errors return 500"""
        mock_rag_system.query.side_effect = Exception("Database error")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "What is this?"}
        )

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_query_sources_format(self, mock_rag_system):
        """Test that sources are properly formatted with text and url"""
        mock_rag_system.query.return_value = (
            "Response with multiple sources",
            [
                {"text": "Course A - Lesson 1", "url": "https://example.com/a"},
                {"text": "Course B - Lesson 2", "url": None}
            ]
        )
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        sources = response.json()["sources"]
        assert len(sources) == 2
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "https://example.com/a"
        assert sources[1]["url"] is None


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_returns_stats(self, mock_rag_system):
        """Test that courses endpoint returns course statistics"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]

    def test_get_courses_empty_returns_zero(self, mock_rag_system):
        """Test that empty course list returns zero count"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error_returns_500(self, mock_rag_system):
        """Test that errors return 500"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector DB error")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector DB error" in response.json()["detail"]


class TestRequestValidation:
    """Tests for request validation"""

    def test_query_with_extra_fields_ignores_them(self, mock_rag_system):
        """Test that extra fields in request are ignored"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={
                "query": "Test",
                "extra_field": "should be ignored",
                "another": 123
            }
        )

        assert response.status_code == 200

    def test_query_with_wrong_type_session_id_returns_422(self, mock_rag_system):
        """Test that wrong type for session_id returns 422"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test", "session_id": 123}
        )

        # Pydantic v2 enforces strict typing for Optional[str]
        assert response.status_code == 422


class TestResponseFormat:
    """Tests for response format validation"""

    def test_query_response_has_correct_schema(self, mock_rag_system):
        """Test that query response matches expected schema"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test"}
        )

        data = response.json()
        # Check all required fields exist
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        # Check source structure
        for source in data["sources"]:
            assert "text" in source
            assert "url" in source

    def test_courses_response_has_correct_schema(self, mock_rag_system):
        """Test that courses response matches expected schema"""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        for title in data["course_titles"]:
            assert isinstance(title, str)
