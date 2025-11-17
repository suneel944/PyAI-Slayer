# PyAI-Slayer Dashboard

Real-time AI Testing Metrics & Reporting Dashboard

## Overview

The PyAI-Slayer Dashboard provides comprehensive visualization of AI testing metrics with real-time updates and historical trend analysis. It features detailed failure analysis with root cause detection and actionable recommendations.

## Features

### 📊 **Real-Time Metrics**
- Live test execution status
- Current pass/fail rates
- Active test monitoring via WebSocket
- Validation metrics tracking

### 📈 **Historical Trends**
- Time-series analysis (24h, 7d, 30d)
- Test success rate trends
- Performance metrics over time
- Language-specific breakdowns

### 🧪 **Test Explorer**
- Filter tests by status, language, type
- Search and sort capabilities
- Detailed test information
- Artifact links (screenshots, traces)

### ❌ **Failed Tests Analysis**
Complete failure analysis with:
- **Prompt/Query** - Original test input
- **Expected vs Actual Response** - Side-by-side comparison
- **Validation Scores** - All metrics with pass/fail indicators
- **Quality Checks** - Comprehensive quality assessment
- **Root Cause Analysis** - Intelligent failure pattern analysis
- **Recommendations** - Actionable suggestions
- **Failure Patterns** - Statistical analysis

### 🔍 **Auto Failure Analysis**
Smart pattern detection:
- Fallback message detection
- Semantic mismatch identification
- Response quality issues
- Language mismatches
- Performance degradation

## Installation

### 1. Install Dependencies

```bash
# Using pip
pip install fastapi uvicorn[standard] websockets

# Or using make (recommended)
make install-dev
```

Dependencies are already included in `pyproject.toml`.

### 2. Verify Installation

```bash
# Check if dashboard module is available
python -c "from dashboard import create_app; print('✓ Dashboard installed')"
```

## Usage

### Start Dashboard

```bash
# Default: http://localhost:8080
make dashboard

# Or directly with Python
python scripts/run_dashboard.py

# Custom host/port
python scripts/run_dashboard.py --host 0.0.0.0 --port 8888

# Or with make
make dashboard-custom HOST=0.0.0.0 PORT=8888
```

### Run Tests with Dashboard

1. **Start the dashboard** in one terminal:
   ```bash
   make dashboard
   ```

2. **Run tests** in another terminal:
   ```bash
   # Ensure Prometheus metrics are enabled
   export ENABLE_PROMETHEUS_METRICS=true
   
   # Run tests
   make test
   ```

3. **View results** in browser:
   ```
   http://localhost:8080
   ```

The dashboard will automatically:
- Collect test execution data
- Store validation details
- Analyze failures
- Display real-time updates

## Dashboard Sections

### 1. Overview Tab

**Real-time metrics cards:**
- Total Tests
- Pass Rate
- Failed Tests
- Average Duration

**Charts:**
- Test Results Distribution (Doughnut)
- Validation Metrics (Bar)

### 2. Test Explorer Tab

**Features:**
- Filter by status (passed/failed/skipped)
- Filter by language (EN/AR/multilingual)
- Time range selection (24h/7d/30d)
- Sortable table view

**Columns:**
- Test Name
- Status Badge
- Language
- Test Type
- Duration
- Timestamp

### 3. Failed Tests Tab

**Failure List:**
- Quick overview of all failed tests
- Language and type filtering
- Click to view detailed analysis

**Detailed Failure View:**
- 📝 **Prompt/Query** - Full test input
- ✅ **Expected Response** - Reference response (if available)
- ❌ **Actual Response** - Model's actual output
- 📊 **Validation Scores**:
  - Semantic Similarity
  - BERTScore F1
  - ROUGE-L F1
  - Threshold comparisons
- 🔍 **Quality Checks**:
  - Minimum length
  - Proper ending
  - No HTML tags
  - Character encoding
- 🔍 **Failure Analysis**:
  - Root cause identification
  - Category classification
  - Detected patterns
  - 💡 Recommendations
- 📎 **Artifacts**:
  - Screenshots
  - Playwright traces
  - Console logs
  - HAR files

**Failure Patterns Chart:**
- Bar chart showing failure distribution
- Categories: Semantic Mismatch, Knowledge Gap, Quality Issue, etc.

### 4. Trends Tab

**Time-series visualization:**
- Test success rate over time
- Validation metrics history
- Performance trends
- Language-specific analysis

## Architecture

```
┌─────────────────────────────────────────────┐
│  Frontend (dashboard.html)                  │
│  • Real-time updates via WebSocket          │
│  • Chart.js visualizations                  │
│  • Dark theme UI                            │
└──────────────┬──────────────────────────────┘
               │ HTTP/WebSocket
┌──────────────▼──────────────────────────────┐
│  Backend API (FastAPI)                      │
│  • REST endpoints                           │
│  • WebSocket for live updates               │
│  • Data aggregation                         │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Data Layer                                 │
│  • SQLite (reports/dashboard.db)            │
│  • Prometheus metrics (in-memory)           │
│  • Test artifacts (filesystem)              │
└─────────────────────────────────────────────┘
```

## API Endpoints

### Metrics
- `GET /api/metrics/current` - Current Prometheus metrics
- `GET /api/metrics/history/{metric_type}` - Historical metrics

### Tests
- `GET /api/tests` - All tests (with filters)
- `GET /api/tests/failed` - Failed tests only
- `GET /api/tests/failed/{test_id}/details` - Detailed failure info
- `GET /api/tests/failed/patterns` - Failure pattern statistics

### Statistics
- `GET /api/statistics` - Overall test statistics

### Artifacts
- `GET /api/artifacts/{artifact_type}/{test_id}` - Serve test artifacts

### Real-time
- `WS /ws` - WebSocket for live updates

## Data Storage

### SQLite Database Schema

**Location:** `reports/dashboard.db`

**Tables:**
- `test_results` - Test execution records
- `validation_details` - Validation data (query/response)
- `scoring_details` - Individual metric scores
- `quality_checks` - Quality check results
- `failure_analysis` - Failure analysis derived from test results and error patterns
- `metrics_snapshots` - Historical metrics
- `test_artifacts` - Artifact file references

### Automatic Data Collection

Data is collected automatically via `conftest.py` hooks:
- Test results captured on completion
- Validation data from AIResponseValidator
- Artifacts discovered and linked
- Failure analysis generated for failed tests

## Configuration

### Environment Variables

```bash
# Enable Prometheus metrics (required for real-time data)
ENABLE_PROMETHEUS_METRICS=true

# Prometheus port (default: 8000)
PROMETHEUS_PORT=8000
```

### Dashboard Settings

Modify in `scripts/run_dashboard.py`:
```python
# Default host and port
HOST = "0.0.0.0"
PORT = 8080
```

## Failure Analysis

### Auto-Detection Patterns

The dashboard automatically detects:

1. **Fallback Messages**
   - "Sorry, I didn't understand"
   - "Try again"
   - Arabic equivalents

2. **Semantic Mismatches**
   - Very low similarity (< 0.3)
   - Below threshold similarity

3. **Quality Issues**
   - Short/empty responses
   - HTML in response
   - Improper endings
   - Low BERTScore/ROUGE

4. **Language Issues**
   - Language mismatches
   - Encoding problems

### Recommendations Engine

Provides actionable suggestions:
- "Review prompt engineering"
- "Check RAG retrieval"
- "Verify model knowledge domain"
- "Adjust similarity threshold"
- "Review expected response realism"

## Tips & Best Practices

### For Real-Time Monitoring

1. Keep dashboard open during test runs
2. Monitor WebSocket connection status (green dot)
3. Use filters to focus on specific test types
4. Refresh data manually if needed

### For Failure Analysis

1. Review failure patterns first
2. Drill down into specific failures
3. Check recommendations
4. Compare expected vs actual responses
5. Review validation scores
6. Check artifacts for visual confirmation

### For Historical Analysis

1. Use Trends tab for long-term patterns
2. Compare different time periods
3. Track improvement over time
4. Identify regression points

## Troubleshooting

### Dashboard Not Starting

```bash
# Check if port is available
lsof -i :8080

# Try different port
python scripts/run_dashboard.py --port 8081

# Check dependencies
pip install fastapi uvicorn websockets
```

### No Data Showing

```bash
# Ensure Prometheus metrics enabled
export ENABLE_PROMETHEUS_METRICS=true

# Run tests to generate data
make test

# Check database
ls -lh reports/dashboard.db
```

### WebSocket Not Connecting

- Check browser console for errors
- Verify firewall settings
- Try `http://127.0.0.1:8080` instead of `localhost`
- Check if WebSocket port is accessible

### Failed Tests Not Showing Details

- Ensure tests have validation data
- Check `conftest.py` integration
- Verify SQLite database permissions
- Look for error logs in terminal

## Development

### Adding Custom Metrics

```python
# In your test
from dashboard.collectors import get_dashboard_collector

collector = get_dashboard_collector()

# Collect custom scoring metric
collector.collect_scoring_metrics(
    test_id=test_id,
    metrics={"custom_score": 0.95},
    thresholds={"custom_score": 0.9}
)
```

### Extending Failure Analysis

Edit `src/dashboard/failure_analyzer.py`:

```python
class FailureAnalyzer:
    def analyze_failure(self, ...):
        # Add custom pattern detection
        if custom_condition:
            patterns.append("custom_pattern")
            recommendations.append("Custom recommendation")
```

### Customizing Dashboard UI

Edit `src/dashboard/static/dashboard.html`:
- Modify CSS in `<style>` section
- Add custom JavaScript functions
- Extend API calls

## Support

For issues or questions:
- Check logs: Dashboard terminal output
- Review database: `sqlite3 reports/dashboard.db`
- Enable debug logging: `LOG_LEVEL=DEBUG`

## Future Enhancements

Planned features:
- Export reports to PDF
- Email notifications for failures
- Slack/Teams integration
- Custom alerting rules
- A/B test comparison
- Model version tracking
- Advanced filtering options
- Multi-project support
