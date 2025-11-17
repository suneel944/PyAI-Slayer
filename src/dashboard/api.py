"""FastAPI backend for AI testing dashboard."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from core.observability import get_prometheus_metrics

from .collectors import get_dashboard_collector
from .data_store import DashboardDataStore
from .models import DashboardMetrics, FailedTestDetail, TestResult


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


def create_app(data_store: DashboardDataStore | None = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="PyAI-Slayer Dashboard API",
        description="Real-time AI testing metrics and reporting dashboard",
        version="1.0.0",
    )

    # Initialize components
    store = data_store or DashboardDataStore()
    collector = get_dashboard_collector()
    manager = ConnectionManager()

    # Mount static files directory if it exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main dashboard HTML."""
        dashboard_file = static_dir / "dashboard.html"
        if dashboard_file.exists():
            return FileResponse(dashboard_file)
        return HTMLResponse(
            "<h1>PyAI-Slayer Dashboard</h1><p>Dashboard UI not found. Check static/dashboard.html</p>"
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/api/metrics/current", response_model=DashboardMetrics)
    async def get_current_metrics():
        """Get current real-time metrics from Prometheus."""
        try:
            prometheus = get_prometheus_metrics()
            if not prometheus.enabled:
                raise HTTPException(status_code=503, detail="Prometheus metrics not enabled")

            summary = prometheus.get_metrics_summary()

            return DashboardMetrics(
                timestamp=datetime.now(),
                tests=summary.get("tests", {}),
                validations=summary.get("validations", {}),
                a_tier_metrics={},  # Could be enhanced to extract A-tier metrics
                browser_operations=summary.get("browser_operations", {}),
            )
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/metrics/history/{metric_type}")
    async def get_metrics_history(metric_type: str, hours: int = 24):
        """Get historical metrics."""
        try:
            history = store.get_metrics_history(metric_type, hours)
            return {"metric_type": metric_type, "hours": hours, "data": history}
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/tests", response_model=list[TestResult])
    async def get_tests(
        status: str | None = None,
        language: str | None = None,
        hours: int = 24,
        limit: int = 100,
    ):
        """Get test results with filters."""
        try:
            date_from = datetime.now() - timedelta(hours=hours)
            tests = store.get_test_results(
                status=status, language=language, date_from=date_from, limit=limit
            )
            return tests
        except Exception as e:
            logger.error(f"Failed to get tests: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/tests/failed", response_model=list[TestResult])
    async def get_failed_tests(language: str | None = None, hours: int = 24, limit: int = 100):
        """Get failed tests."""
        try:
            date_from = datetime.now() - timedelta(hours=hours)
            tests = store.get_test_results(
                status="failed", language=language, date_from=date_from, limit=limit
            )
            return tests
        except Exception as e:
            logger.error(f"Failed to get failed tests: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/tests/failed/{test_id}/details", response_model=FailedTestDetail)
    async def get_failed_test_details(test_id: str):
        """Get detailed failed test information."""
        try:
            details = store.get_failed_test_detail(test_id)
            if not details:
                raise HTTPException(status_code=404, detail="Test not found")
            return details
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get test details: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/tests/failed/patterns")
    async def get_failure_patterns():
        """Get failure pattern statistics."""
        try:
            patterns = store.get_failure_patterns()
            return patterns
        except Exception as e:
            logger.error(f"Failed to get failure patterns: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/statistics")
    async def get_statistics():
        """Get overall test statistics."""
        try:
            # Collect system metrics on-demand when statistics are requested
            # This ensures we have current system resource data
            metrics_collected = 0
            try:
                metrics_collected = collector.collect_system_metrics()
                if metrics_collected > 0:
                    logger.debug(f"Collected {metrics_collected} system metrics")
            except Exception as e:
                logger.warning(f"Could not collect system metrics: {e}", exc_info=True)

            # Get statistics (includes system metrics from database)
            stats = store.get_test_statistics()
            system_metrics = stats.get("system_metrics", {})
            system_metrics_keys = list(system_metrics.keys())

            # Log system metrics availability
            if system_metrics_keys:
                logger.debug(f"System metrics available: {system_metrics_keys}")
            else:
                if metrics_collected == 0:
                    logger.debug("No system metrics collected or available")
                else:
                    logger.warning(
                        f"⚠️ {metrics_collected} metric(s) were collected but not found in database. "
                        "This may indicate a database transaction issue."
                    )

            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/trends")
    async def get_trends(hours: int = 168):
        """Get trend data for charts."""
        try:
            trends = store.get_trends_data(hours)
            return trends
        except Exception as e:
            logger.error(f"Failed to get trends: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await manager.connect(websocket)
        try:
            while True:
                # Check if connection is still open before sending
                try:
                    if websocket.client_state.value != 1:  # 1 = CONNECTED
                        break
                except Exception:
                    break

                # Send heartbeat/metrics every 5 seconds
                try:
                    # Try to get Prometheus metrics, but don't fail if unavailable
                    metrics_sent = False
                    try:
                        prometheus = get_prometheus_metrics()
                        if prometheus and prometheus.enabled:
                            summary = prometheus.get_metrics_summary()
                            # Only send if we have valid data
                            if summary and isinstance(summary, dict) and summary.get("tests"):
                                await websocket.send_json(
                                    {
                                        "type": "metrics_update",
                                        "timestamp": datetime.now().isoformat(),
                                        "data": summary,
                                    }
                                )
                                metrics_sent = True
                    except Exception as prom_error:
                        logger.debug(f"Could not get Prometheus metrics: {prom_error}")

                    # If no metrics sent, send heartbeat to keep connection alive
                    if not metrics_sent:
                        await websocket.send_json(
                            {
                                "type": "heartbeat",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                except WebSocketDisconnect:
                    break
                except Exception:
                    # Connection might be broken, exit gracefully
                    break

                await asyncio.sleep(5)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            manager.disconnect(websocket)

    @app.post("/api/collect/test")
    async def collect_test_result(
        test_name: str,
        status: str,
        duration: float,
        language: str = "unknown",
        test_type: str = "general",
        test_path: str | None = None,
    ):
        """Endpoint for collecting test results."""
        try:
            from .collectors import should_exclude_test

            # Check if test should be excluded (unit/integration tests)
            if should_exclude_test(test_name, test_path):
                logger.debug(
                    f"Test {test_name} excluded from dashboard data collection "
                    "(unit/integration test)"
                )
                return {"test_id": None, "status": "excluded", "reason": "unit/integration test"}

            test_id = collector.collect_test_result(
                test_name=test_name,
                status=status,
                duration=duration,
                language=language,
                test_type=test_type,
                test_path=test_path,
            )

            # Only broadcast if test was not excluded
            if test_id is not None:
                # Broadcast to connected clients
                await manager.broadcast(
                    {
                        "type": "test_completed",
                        "test_id": test_id,
                        "test_name": test_name,
                        "status": status,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            return {"test_id": test_id, "status": "collected" if test_id else "excluded"}
        except Exception as e:
            logger.error(f"Failed to collect test result: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/artifacts/{artifact_type}/{test_id}")
    async def get_artifact(artifact_type: str, test_id: str):
        """Serve test artifact (screenshot, trace, etc)."""
        try:
            # Query database for artifact path
            with store.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT file_path FROM test_artifacts
                    WHERE test_id = ? AND artifact_type = ?
                    ORDER BY timestamp DESC LIMIT 1
                """,
                    (test_id, artifact_type),
                )
                row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Artifact not found")

            file_path = Path(row["file_path"])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Artifact file not found on disk")

            return FileResponse(file_path)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to serve artifact: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):  # nosec B104
    """Run dashboard server."""
    import uvicorn

    app = create_app()
    logger.info(f"Starting dashboard server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
