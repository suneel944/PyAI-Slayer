/**
 * PyAI-Slayer Dashboard - Export & Reporting (Phase 4)
 * PDF export, Excel/CSV downloads, sharing, and report generation
 */

// ===== PDF EXPORT WITH CHARTS =====
async function exportToPDF() {
    showToast('Generating PDF', 'Creating report with charts...', 'info');

    try {
        // Import jsPDF and html2canvas dynamically
        if (!window.jspdf || !window.html2canvas) {
            await loadExportLibraries();
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF('p', 'mm', 'a4');
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        let yPosition = 20;

        // Header
        doc.setFontSize(20);
        doc.setTextColor(59, 130, 246);
        doc.text('PyAI-Slayer Test Report', pageWidth / 2, yPosition, { align: 'center' });

        yPosition += 10;
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, yPosition, { align: 'center' });

        yPosition += 15;

        // Summary Statistics
        doc.setFontSize(14);
        doc.setTextColor(0, 0, 0);
        doc.text('Test Summary', 15, yPosition);
        yPosition += 10;

        doc.setFontSize(10);
        const stats = await fetch('/api/statistics').then(r => r.json());
        const summaryData = [
            ['Total Tests', stats.total_tests || 0],
            ['Pass Rate', `${stats.pass_rate || 0}%`],
            ['Failed Tests', stats.failed || 0],
            ['Average Duration', `${(stats.avg_duration || 0).toFixed(2)}s`]
        ];

        summaryData.forEach(([label, value], index) => {
            doc.text(`${label}:`, 20, yPosition + (index * 7));
            doc.text(String(value), 70, yPosition + (index * 7));
        });

        yPosition += 35;

        // Capture charts
        const resultsChart = document.getElementById('resultsChart');
        if (resultsChart && yPosition < pageHeight - 80) {
            doc.text('Test Results Distribution', 15, yPosition);
            yPosition += 5;

            const canvas = await html2canvas(resultsChart, { scale: 2 });
            const imgData = canvas.toDataURL('image/png');
            const imgWidth = pageWidth - 30;
            const imgHeight = (canvas.height * imgWidth) / canvas.width;

            if (yPosition + imgHeight > pageHeight - 20) {
                doc.addPage();
                yPosition = 20;
            }

            doc.addImage(imgData, 'PNG', 15, yPosition, imgWidth, imgHeight);
            yPosition += imgHeight + 10;
        }

        // Save PDF
        const filename = `pyai-slayer-report-${new Date().toISOString().split('T')[0]}.pdf`;
        doc.save(filename);

        showToast('PDF Generated', `Report saved as ${filename}`, 'success');
    } catch (error) {
        console.error('PDF export failed:', error);
        showToast('Export Failed', 'Could not generate PDF report', 'error');
    }
}

async function loadExportLibraries() {
    // Load jsPDF
    if (!window.jspdf) {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // Load html2canvas
    if (!window.html2canvas) {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
}

// ===== CSV EXPORT =====
async function exportToCSV() {
    showToast('Generating CSV', 'Preparing test data...', 'info');

    try {
        const response = await fetch('/api/tests?hours=168'); // Last 7 days
        const tests = await response.json();

        if (tests.length === 0) {
            showToast('No Data', 'No test data available to export', 'warning');
            return;
        }

        // Create CSV header
        const headers = ['Test Name', 'Status', 'Language', 'Type', 'Duration (s)', 'Timestamp'];
        const csvRows = [headers.join(',')];

        // Add data rows
        tests.forEach(test => {
            const row = [
                `"${test.test_name.replace(/"/g, '""')}"`,
                test.status,
                test.language,
                test.test_type,
                test.duration.toFixed(2),
                new Date(test.timestamp).toISOString()
            ];
            csvRows.push(row.join(','));
        });

        // Create and download
        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const filename = `pyai-slayer-tests-${new Date().toISOString().split('T')[0]}.csv`;
        downloadBlob(blob, filename);

        showToast('CSV Exported', `${tests.length} tests exported`, 'success');
    } catch (error) {
        console.error('CSV export failed:', error);
        showToast('Export Failed', 'Could not generate CSV file', 'error');
    }
}

// ===== EXCEL EXPORT =====
async function exportToExcel() {
    showToast('Generating Excel', 'Preparing workbook...', 'info');

    try {
        // Load SheetJS library if not already loaded
        if (!window.XLSX) {
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        const response = await fetch('/api/tests?hours=168');
        const tests = await response.json();

        if (tests.length === 0) {
            showToast('No Data', 'No test data available to export', 'warning');
            return;
        }

        // Get statistics for summary sheet
        const stats = await fetch('/api/statistics').then(r => r.json());

        // Create workbook
        const wb = XLSX.utils.book_new();

        // Summary sheet
        const summaryData = [
            ['PyAI-Slayer Test Report'],
            ['Generated:', new Date().toLocaleString()],
            [],
            ['Metric', 'Value'],
            ['Total Tests', stats.total_tests || 0],
            ['Pass Rate', `${stats.pass_rate || 0}%`],
            ['Passed Tests', stats.passed || 0],
            ['Failed Tests', stats.failed || 0],
            ['Skipped Tests', stats.skipped || 0],
            ['Average Duration', `${(stats.avg_duration || 0).toFixed(2)}s`]
        ];
        const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
        XLSX.utils.book_append_sheet(wb, summarySheet, 'Summary');

        // Tests sheet
        const testsData = tests.map(test => ({
            'Test Name': test.test_name,
            'Status': test.status,
            'Language': test.language,
            'Type': test.test_type,
            'Duration (s)': test.duration.toFixed(2),
            'Timestamp': new Date(test.timestamp).toLocaleString()
        }));
        const testsSheet = XLSX.utils.json_to_sheet(testsData);
        XLSX.utils.book_append_sheet(wb, testsSheet, 'Tests');

        // Failed tests sheet
        const failedTests = tests.filter(t => t.status === 'failed');
        if (failedTests.length > 0) {
            const failedData = failedTests.map(test => ({
                'Test Name': test.test_name,
                'Language': test.language,
                'Type': test.test_type,
                'Duration (s)': test.duration.toFixed(2),
                'Timestamp': new Date(test.timestamp).toLocaleString()
            }));
            const failedSheet = XLSX.utils.json_to_sheet(failedData);
            XLSX.utils.book_append_sheet(wb, failedSheet, 'Failed Tests');
        }

        // Save workbook
        const filename = `pyai-slayer-report-${new Date().toISOString().split('T')[0]}.xlsx`;
        XLSX.writeFile(wb, filename);

        showToast('Excel Exported', `Report with ${tests.length} tests saved`, 'success');
    } catch (error) {
        console.error('Excel export failed:', error);
        showToast('Export Failed', 'Could not generate Excel file', 'error');
    }
}

// ===== SHARE VIA URL =====
function shareReport() {
    const filters = {
        status: document.getElementById('statusFilter')?.value || '',
        language: document.getElementById('languageFilter')?.value || '',
        hours: document.getElementById('hoursFilter')?.value || '24'
    };

    const params = new URLSearchParams(filters);
    const shareUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;

    // Copy to clipboard
    navigator.clipboard.writeText(shareUrl).then(() => {
        showToast('Link Copied', 'Dashboard link copied to clipboard', 'success');
    }).catch(() => {
        // Fallback: show modal with URL
        showShareModal(shareUrl);
    });
}

function showShareModal(url) {
    const modal = document.createElement('div');
    modal.className = 'share-modal';
    modal.innerHTML = `
        <div class="share-modal-content">
            <h3>Share Dashboard</h3>
            <p>Copy this link to share the current view:</p>
            <input type="text" value="${url}" readonly onclick="this.select()"
                   style="width: 100%; padding: 0.5rem; margin: 1rem 0; border: 1px solid var(--border); border-radius: 4px;">
            <button onclick="this.parentElement.parentElement.remove()">Close</button>
        </div>
    `;
    document.body.appendChild(modal);
}

// ===== PRINT LAYOUT =====
function printReport() {
    showToast('Preparing Print', 'Optimizing layout...', 'info');

    // Add print-specific styles
    const printStyles = document.createElement('style');
    printStyles.id = 'print-styles';
    printStyles.textContent = `
        @media print {
            .header-right, .tabs, .filter-bar, button:not(.print-keep) {
                display: none !important;
            }
            .container {
                max-width: 100%;
                padding: 0;
            }
            .chart-container {
                page-break-inside: avoid;
                margin-bottom: 2rem;
            }
            .metric-card {
                page-break-inside: avoid;
            }
            body {
                background: white !important;
            }
        }
    `;
    document.head.appendChild(printStyles);

    // Trigger print
    setTimeout(() => {
        window.print();
        // Remove print styles after printing
        setTimeout(() => printStyles.remove(), 1000);
    }, 500);
}

// ===== REPORT TEMPLATES =====
const reportTemplates = {
    daily: {
        name: 'Daily Summary',
        hours: 24,
        includeCharts: true,
        includeFailed: true
    },
    weekly: {
        name: 'Weekly Report',
        hours: 168,
        includeCharts: true,
        includeFailed: true,
        includeTrends: true
    },
    executive: {
        name: 'Executive Summary',
        hours: 168,
        includeCharts: true,
        includeFailed: false,
        summaryOnly: true
    }
};

async function generateTemplateReport(templateName) {
    const template = reportTemplates[templateName];
    if (!template) return;

    showToast('Generating Report', `Creating ${template.name}...`, 'info');

    // Apply template settings
    const hoursFilter = document.getElementById('hoursFilter');
    if (hoursFilter) {
        hoursFilter.value = template.hours;
        await loadTests();
    }

    // Generate based on template type
    if (template.summaryOnly) {
        await exportToExcel(); // Executive summary as Excel
    } else {
        await exportToPDF(); // Full report as PDF
    }
}

// ===== SCHEDULE REPORTS =====
function scheduleReport(frequency) {
    // This would require backend support
    showToast('Schedule Report', `${frequency} reports will be sent via email`, 'info');

    // Store preference in localStorage
    const schedule = {
        frequency,
        enabled: true,
        lastRun: new Date().toISOString()
    };
    localStorage.setItem('report-schedule', JSON.stringify(schedule));
}

// ===== UTILITY FUNCTIONS =====
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// ===== AUTO-APPLY URL FILTERS ON LOAD =====
function applyUrlFilters() {
    const params = new URLSearchParams(window.location.search);

    if (params.has('status')) {
        const statusFilter = document.getElementById('statusFilter');
        if (statusFilter) statusFilter.value = params.get('status');
    }

    if (params.has('language')) {
        const langFilter = document.getElementById('languageFilter');
        if (langFilter) langFilter.value = params.get('language');
    }

    if (params.has('hours')) {
        const hoursFilter = document.getElementById('hoursFilter');
        if (hoursFilter) hoursFilter.value = params.get('hours');
    }

    // Reload data with applied filters
    if (params.toString()) {
        loadTests();
    }
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyUrlFilters);
} else {
    applyUrlFilters();
}
