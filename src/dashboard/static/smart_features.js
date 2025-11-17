/**
 * PyAI-Slayer Dashboard - Smart Features & Analytics (Phase 3)
 * Advanced search, filters, and analytics capabilities
 */

// ===== ADVANCED SEARCH WITH FUZZY MATCHING =====
function fuzzySearch(text, searchTerm) {
    if (!searchTerm) return true;

    const textLower = text.toLowerCase();
    const searchLower = searchTerm.toLowerCase();

    // Exact match
    if (textLower.includes(searchLower)) return true;

    // Fuzzy matching - allow for typos
    let searchIndex = 0;
    for (let i = 0; i < textLower.length && searchIndex < searchLower.length; i++) {
        if (textLower[i] === searchLower[searchIndex]) {
            searchIndex++;
        }
    }
    return searchIndex === searchLower.length;
}

function applySearch(searchTerm) {
    const rows = document.querySelectorAll('#testsTableBody tr, #failedTestsTableBody tr');
    let visibleCount = 0;

    rows.forEach(row => {
        const text = row.textContent;
        const matches = fuzzySearch(text, searchTerm);
        row.style.display = matches ? '' : 'none';
        if (matches) visibleCount++;
    });

    // Show count
    updateSearchResults(visibleCount);
}

function updateSearchResults(count) {
    const container = document.getElementById('searchResults');
    if (container) {
        container.textContent = count > 0 ? `${count} result${count !== 1 ? 's' : ''}` : 'No results';
    }
}

// ===== MULTI-SELECT FILTERS WITH CHIPS =====
let activeFilters = {
    status: [],
    language: [],
    type: []
};

function toggleFilter(category, value) {
    const index = activeFilters[category].indexOf(value);
    if (index > -1) {
        activeFilters[category].splice(index, 1);
    } else {
        activeFilters[category].push(value);
    }
    updateFilterChips();
    applyFilters();
}

function updateFilterChips() {
    const container = document.getElementById('filterChips');
    if (!container) return;

    container.innerHTML = '';

    Object.keys(activeFilters).forEach(category => {
        activeFilters[category].forEach(value => {
            const chip = document.createElement('div');
            chip.className = 'filter-chip';
            chip.innerHTML = `
                <span>${formatFilterLabel(category, value)}</span>
                <button onclick="removeFilter('${category}', '${value}')" class="chip-remove">✕</button>
            `;
            container.appendChild(chip);
        });
    });

    // Add clear all button if any filters active
    const hasFilters = Object.values(activeFilters).some(arr => arr.length > 0);
    if (hasFilters) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'filter-chip-clear';
        clearBtn.textContent = 'Clear All';
        clearBtn.onclick = clearAllFilters;
        container.appendChild(clearBtn);
    }
}

function removeFilter(category, value) {
    const index = activeFilters[category].indexOf(value);
    if (index > -1) {
        activeFilters[category].splice(index, 1);
    }
    updateFilterChips();
    applyFilters();
}

function clearAllFilters() {
    activeFilters = { status: [], language: [], type: [] };
    updateFilterChips();
    applyFilters();
}

function formatFilterLabel(category, value) {
    const labels = {
        status: { passed: '✓ Passed', failed: '✗ Failed', skipped: '⊘ Skipped' },
        language: { en: '🇬🇧 EN', ar: '🇸🇦 AR', multilingual: '🌐 Multi' },
        type: {}
    };
    return labels[category][value] || value;
}

function applyFilters() {
    // Will be implemented with actual filtering logic
    showToast('Filters Applied', 'Updated test results', 'info');
    loadTests(); // Reload with filters
}

// ===== FILTER PRESETS =====
let filterPresets = JSON.parse(localStorage.getItem('dashboard-filter-presets')) || {};

function saveFilterPreset() {
    const name = prompt('Enter preset name:');
    if (!name) return;

    filterPresets[name] = {
        ...activeFilters,
        saved: new Date().toISOString()
    };
    localStorage.setItem('dashboard-filter-presets', JSON.stringify(filterPresets));
    updatePresetsList();
    showToast('Preset Saved', `"${name}" saved successfully`, 'success');
}

function loadFilterPreset(name) {
    if (!filterPresets[name]) return;

    activeFilters = {
        status: [...(filterPresets[name].status || [])],
        language: [...(filterPresets[name].language || [])],
        type: [...(filterPresets[name].type || [])]
    };
    updateFilterChips();
    applyFilters();
    showToast('Preset Loaded', `Loaded "${name}"`, 'info');
}

function deleteFilterPreset(name) {
    if (!confirm(`Delete preset "${name}"?`)) return;

    delete filterPresets[name];
    localStorage.setItem('dashboard-filter-presets', JSON.stringify(filterPresets));
    updatePresetsList();
    showToast('Preset Deleted', `"${name}" removed`, 'info');
}

function updatePresetsList() {
    const container = document.getElementById('filterPresets');
    if (!container) return;

    const presetNames = Object.keys(filterPresets);
    if (presetNames.length === 0) {
        container.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.875rem;">No saved presets</div>';
        return;
    }

    container.innerHTML = presetNames.map(name => `
        <div class="preset-item">
            <button onclick="loadFilterPreset('${name}')" class="preset-load">${name}</button>
            <button onclick="deleteFilterPreset('${name}')" class="preset-delete">🗑️</button>
        </div>
    `).join('');
}

// ===== DATE RANGE PICKER =====
let dateRange = {
    start: null,
    end: null
};

function setDateRange(preset) {
    const now = new Date();
    const start = new Date();

    switch (preset) {
        case 'today':
            start.setHours(0, 0, 0, 0);
            break;
        case 'yesterday':
            start.setDate(start.getDate() - 1);
            start.setHours(0, 0, 0, 0);
            now.setDate(now.getDate() - 1);
            now.setHours(23, 59, 59, 999);
            break;
        case 'week':
            start.setDate(start.getDate() - 7);
            break;
        case 'month':
            start.setMonth(start.getMonth() - 1);
            break;
        case 'custom':
            // Custom range will be set via date inputs
            return;
    }

    dateRange = { start, end: now };
    updateDateRangeDisplay();
    applyFilters();
}

function updateDateRangeDisplay() {
    const display = document.getElementById('dateRangeDisplay');
    if (!display) return;

    if (!dateRange.start || !dateRange.end) {
        display.textContent = 'All Time';
        return;
    }

    const format = date => date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    display.textContent = `${format(dateRange.start)} - ${format(dateRange.end)}`;
}

// ===== TIME PERIOD COMPARISON =====
let comparisonPeriod = null;

function enableComparison(enabled) {
    comparisonPeriod = enabled ? 'previous' : null;
    if (enabled) {
        calculateComparison();
    } else {
        clearComparison();
    }
}

async function calculateComparison() {
    try {
        // Get current period stats
        const currentStats = await fetch('/api/statistics').then(r => r.json());

        // Get previous period (same duration, shifted back)
        const hours = 24; // Default to 24h comparison
        const prevParams = new URLSearchParams({ hours: hours * 2 });
        const historicalData = await fetch(`/api/tests?${prevParams}`).then(r => r.json());

        // Calculate comparison metrics
        const comparison = {
            tests: {
                current: currentStats.total_tests,
                change: calculateChange(currentStats.total_tests, historicalData.length / 2)
            },
            passRate: {
                current: currentStats.pass_rate,
                change: calculateChange(currentStats.pass_rate, 85) // Would need historical calculation
            },
            failures: {
                current: currentStats.failed,
                change: calculateChange(currentStats.failed, historicalData.filter(t => t.status === 'failed').length / 2)
            }
        };

        displayComparison(comparison);
    } catch (error) {
        console.error('Failed to calculate comparison:', error);
    }
}

function calculateChange(current, previous) {
    if (previous === 0) return current > 0 ? 100 : 0;
    return ((current - previous) / previous * 100).toFixed(1);
}

function displayComparison(comparison) {
    // Update metric cards with comparison data
    const cards = {
        testsChange: comparison.tests.change,
        passRateChange: comparison.passRate.change,
        failedChange: comparison.failures.change
    };

    Object.keys(cards).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            const change = cards[id];
            const isPositive = id === 'passRateChange' ? change > 0 : change < 0;
            element.className = `metric-change ${isPositive ? 'positive' : change < 0 ? 'negative' : 'neutral'}`;
            element.innerHTML = `<span>${isPositive ? '↑' : '↓'} ${Math.abs(change)}% vs previous</span>`;
        }
    });
}

function clearComparison() {
    // Reset to default display
    document.getElementById('testsChange')?.classList.replace('positive', 'neutral');
    document.getElementById('testsChange')?.classList.replace('negative', 'neutral');
    document.getElementById('passRateChange')?.classList.replace('negative', 'positive');
    document.getElementById('failedChange')?.classList.replace('positive', 'negative');
}

// ===== PERFORMANCE BENCHMARKING =====
function calculatePerformanceBenchmarks(tests) {
    const durations = tests.map(t => t.duration).filter(d => d > 0);
    if (durations.length === 0) return null;

    durations.sort((a, b) => a - b);

    return {
        min: durations[0],
        max: durations[durations.length - 1],
        avg: durations.reduce((a, b) => a + b, 0) / durations.length,
        median: durations[Math.floor(durations.length / 2)],
        p95: durations[Math.floor(durations.length * 0.95)],
        p99: durations[Math.floor(durations.length * 0.99)]
    };
}

function displayBenchmarks(benchmarks) {
    const container = document.getElementById('performanceBenchmarks');
    if (!container || !benchmarks) return;

    container.innerHTML = `
        <div class="benchmark-grid">
            <div class="benchmark-item">
                <div class="benchmark-label">Min</div>
                <div class="benchmark-value">${benchmarks.min.toFixed(2)}s</div>
            </div>
            <div class="benchmark-item">
                <div class="benchmark-label">Avg</div>
                <div class="benchmark-value">${benchmarks.avg.toFixed(2)}s</div>
            </div>
            <div class="benchmark-item">
                <div class="benchmark-label">Median</div>
                <div class="benchmark-value">${benchmarks.median.toFixed(2)}s</div>
            </div>
            <div class="benchmark-item">
                <div class="benchmark-label">P95</div>
                <div class="benchmark-value">${benchmarks.p95.toFixed(2)}s</div>
            </div>
            <div class="benchmark-item">
                <div class="benchmark-label">P99</div>
                <div class="benchmark-value">${benchmarks.p99.toFixed(2)}s</div>
            </div>
            <div class="benchmark-item">
                <div class="benchmark-label">Max</div>
                <div class="benchmark-value">${benchmarks.max.toFixed(2)}s</div>
            </div>
        </div>
    `;
}

// ===== CUSTOM METRIC CALCULATIONS =====
function calculateCustomMetrics(tests) {
    const metrics = {
        totalDuration: tests.reduce((sum, t) => sum + t.duration, 0),
        avgTestsPerHour: tests.length / 24,
        successRate: (tests.filter(t => t.status === 'passed').length / tests.length * 100).toFixed(1),
        languageDistribution: {},
        typeDistribution: {},
        hourlyDistribution: new Array(24).fill(0)
    };

    // Calculate distributions
    tests.forEach(test => {
        metrics.languageDistribution[test.language] = (metrics.languageDistribution[test.language] || 0) + 1;
        metrics.typeDistribution[test.test_type] = (metrics.typeDistribution[test.test_type] || 0) + 1;

        const hour = new Date(test.timestamp).getHours();
        metrics.hourlyDistribution[hour]++;
    });

    return metrics;
}

// ===== SORT FUNCTIONALITY =====
let currentSort = { column: null, direction: 'asc' };

function sortTable(column) {
    const direction = currentSort.column === column && currentSort.direction === 'asc' ? 'desc' : 'asc';
    currentSort = { column, direction };

    const tbody = document.querySelector('#testsTableBody');
    if (!tbody) return;

    const rows = Array.from(tbody.querySelectorAll('tr'));
    const sortedRows = rows.sort((a, b) => {
        const aText = a.cells[getColumnIndex(column)]?.textContent || '';
        const bText = b.cells[getColumnIndex(column)]?.textContent || '';

        const result = aText.localeCompare(bText, undefined, { numeric: true });
        return direction === 'asc' ? result : -result;
    });

    tbody.innerHTML = '';
    sortedRows.forEach(row => tbody.appendChild(row));

    // Update sort indicators
    updateSortIndicators(column, direction);
}

function getColumnIndex(column) {
    const columns = { name: 0, status: 1, language: 2, type: 3, duration: 4, timestamp: 5 };
    return columns[column] || 0;
}

function updateSortIndicators(column, direction) {
    document.querySelectorAll('th').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
    });

    const th = document.querySelector(`th[data-column="${column}"]`);
    if (th) {
        th.classList.add(`sort-${direction}`);
    }
}

// ===== INITIALIZATION =====
function initSmartFeatures() {
    updateFilterChips();
    updatePresetsList();
    updateDateRangeDisplay();
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSmartFeatures);
} else {
    initSmartFeatures();
}
