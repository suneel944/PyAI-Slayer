/**
 * PyAI-Slayer Metrics Hub Dashboard
 * Beautiful metrics dashboard with comprehensive AI testing analytics
 */

// ===== GLOBAL STATE =====
let ws = null;
let charts = {};
let currentTab = 'overview';
let chartAnimationsPlayed = {
    performanceTrend: false,
    healthRadar: false,
    categoryBreakdown: false,
    resourceUtilization: false
};
let metricsData = {
    baseModel: {},
    rag: {},
    safety: {},
    performance: {},
    reliability: {},
    agent: {},
    security: {}
};

// ===== INITIALIZATION =====
// Set dark mode only
document.documentElement.setAttribute('data-theme', 'dark');

document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadAllMetrics();
    setupEventListeners();

    // Load test results to populate the badge count
    loadTestResults();

    // Initialize tab indicator on the default active tab (Overview)
    initializeTabIndicator();

    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Initialize ultra-smooth scrolling with Lenis (60 FPS)
    initSmoothScrolling();

    // After everything is initialized, restore the saved tab if different from default
    setTimeout(() => {
        let savedTab = 'overview'; // Default to overview
        try {
            const storedTab = localStorage.getItem('dashboardActiveTab');
            if (storedTab) {
                savedTab = storedTab;
            }
        } catch (e) {
            console.warn('Could not restore tab from localStorage:', e);
        }

        // If saved tab is different from overview, switch to it
        if (savedTab !== 'overview') {
            switchTab(savedTab);
        }
    }, 100); // Small delay to ensure everything is fully rendered
});

// ===== SMOOTH SCROLLING INITIALIZATION =====
let lenis = null;
let displayRefreshRate = 60; // Default to 60Hz
let targetFPS = 60;

// ===== DISPLAY REFRESH RATE DETECTION =====
function detectDisplayRefreshRate() {
    return new Promise((resolve) => {
        let lastFrameTime = performance.now();
        let frameCount = 0;
        let startTime = lastFrameTime;

        function measureFrame() {
            const currentTime = performance.now();
            frameCount++;

            // Measure for 1 second to get accurate refresh rate
            if (currentTime - startTime >= 1000) {
                const detectedRate = Math.round(frameCount);
                // Cap at reasonable values (60-240Hz)
                const refreshRate = Math.min(Math.max(detectedRate, 60), 240);
                resolve(refreshRate);
                return;
            }

            requestAnimationFrame(measureFrame);
        }

        requestAnimationFrame(measureFrame);
    });
}

// Initialize refresh rate detection
detectDisplayRefreshRate().then(rate => {
    displayRefreshRate = rate;
    targetFPS = rate >= 120 ? 120 : 60; // Target 120 FPS on high refresh displays, 60 otherwise

    // Update Lenis if already initialized
    if (lenis) {
        // Re-initialize with optimal settings for detected refresh rate
        console.log(`🎯 Display refresh rate: ${displayRefreshRate}Hz | Targeting: ${targetFPS} FPS`);
    }
});

function initSmoothScrolling() {
    // Check if Lenis is available
    if (typeof Lenis !== 'undefined') {
        // Detect if we're on a mobile device
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

        // Optimize duration based on target FPS (faster for higher FPS)
        const baseDuration = isMobile ? 1.0 : 1.2;
        const optimizedDuration = targetFPS >= 120 ? baseDuration * 0.9 : baseDuration;

        // Initialize Lenis with ultra-smooth settings optimized for 120 FPS
        lenis = new Lenis({
            duration: optimizedDuration,  // Optimized for high refresh rate
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // Ultra-smooth easing
            orientation: 'vertical',     // Vertical scrolling
            gestureOrientation: 'vertical',
            smoothWheel: true,           // Smooth mouse wheel scrolling (desktop)
            wheelMultiplier: targetFPS >= 120 ? 0.9 : 1,  // Slightly faster on high refresh
            smoothTouch: false,          // Disable on touch devices for native feel
            touchMultiplier: 2,          // Touch sensitivity
            infinite: false,             // Don't allow infinite scrolling
            // Ultra-performance optimizations
            syncTouch: false,            // Don't sync touch events
            // Enable lerp for ultra-smooth interpolation
            lerp: targetFPS >= 120 ? 0.1 : 0.15, // Faster lerp for higher FPS
            // Exclude scrollable containers from Lenis smooth scrolling
            wrapper: window,
            content: document.documentElement,
        });

        // Ultra-high-performance RAF loop optimized for 120 FPS
        let rafId = null;
        let lastTime = 0;
        let frameCount = 0;
        let fpsStartTime = performance.now();

        function ultraSmoothRAF(time) {
            // Calculate delta time for frame-independent updates
            const deltaTime = time - lastTime;
            lastTime = time;

            // Update Lenis
            lenis.raf(time);

            // Frame rate monitoring (every second)
            frameCount++;
            if (time - fpsStartTime >= 1000) {
                const currentFPS = Math.round(frameCount);
                if (currentFPS >= 100) {
                    // Only log if achieving high FPS
                    console.log(`🚀 Scrolling at ${currentFPS} FPS`);
                }
                frameCount = 0;
                fpsStartTime = time;
            }

            // Continue RAF loop with highest priority
            rafId = requestAnimationFrame(ultraSmoothRAF);
        }

        // Start ultra-smooth RAF loop
        rafId = requestAnimationFrame(ultraSmoothRAF);

        // Store rafId for cleanup if needed
        window.lenisRAFId = rafId;

        // Enhance all anchor link scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href !== '#' && href.length > 1) {
                    const target = document.querySelector(href);
                    if (target) {
                        e.preventDefault();
                        lenis.scrollTo(target, {
                            offset: -20, // Account for any fixed headers
                            duration: targetFPS >= 120 ? 1.0 : 1.2, // Optimized for high refresh
                            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // Ultra-smooth
                        });
                    }
                }
            });
        });

        // Expose lenis globally for other modules
        window.lenis = lenis;
        window.displayRefreshRate = displayRefreshRate;
        window.targetFPS = targetFPS;

        // Fix container scrolling: Prevent Lenis from intercepting scroll events in scrollable containers
        function setupContainerScrolling() {
            // Find all scrollable containers
            const scrollableContainers = document.querySelectorAll(
                '.test-results-container, .test-results-list, [class*="-container"]:not([class*="dashboard-container"]), [class*="-list"], [class*="-menu"]:not([class*="custom-dropdown-menu"])'
            );

            scrollableContainers.forEach(container => {
                // Check if container is actually scrollable
                const isScrollable = container.scrollHeight > container.clientHeight;
                if (!isScrollable) return;

                // Prevent Lenis from handling wheel events within this container
                container.addEventListener('wheel', (e) => {
                    // Check if container can scroll in the direction of the wheel event
                    const canScrollUp = container.scrollTop > 0;
                    const canScrollDown = container.scrollTop < container.scrollHeight - container.clientHeight;
                    const scrollingDown = e.deltaY > 0;
                    const scrollingUp = e.deltaY < 0;

                    // If container can scroll in this direction, stop propagation to Lenis
                    if ((scrollingDown && canScrollDown) || (scrollingUp && canScrollUp)) {
                        // Stop event from reaching Lenis
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                    }
                }, { passive: false, capture: true });

                // Also handle touch events for mobile
                container.addEventListener('touchstart', (e) => {
                    // Mark that touch started in this container
                    container.dataset.touching = 'true';
                }, { passive: true });

                container.addEventListener('touchend', () => {
                    delete container.dataset.touching;
                }, { passive: true });

                container.addEventListener('touchmove', (e) => {
                    if (container.dataset.touching === 'true') {
                        e.stopPropagation();
                    }
                }, { passive: true });
            });
        }

        // Setup container scrolling after DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupContainerScrolling);
        } else {
            setupContainerScrolling();
        }

        // Use MutationObserver to re-setup container scrolling when DOM changes
        const observer = new MutationObserver(() => {
            setupContainerScrolling();
        });

        // Observe changes to the test results container
        const testResultsContainer = document.querySelector('.test-results-container');
        if (testResultsContainer) {
            observer.observe(testResultsContainer, {
                childList: true,
                subtree: true
            });
        }

        // Also observe the main content area for other containers
        const mainContent = document.querySelector('.dashboard-content') || document.body;
        observer.observe(mainContent, {
            childList: true,
            subtree: true
        });

        // Log initialization with detected refresh rate
        setTimeout(() => {
            console.log(`✅ Ultra-smooth scrolling initialized with Lenis`);
            console.log(`📊 Display: ${displayRefreshRate}Hz | Target: ${targetFPS} FPS | Ultra-butter-smooth mode enabled`);
        }, 100);
    } else {
        // Fallback: Use native scrolling
        console.warn('⚠️ Lenis not available, using native scrolling');
    }
}

function setupEventListeners() {
    // Initialize custom dropdown
    initCustomDropdown();

    // Update indicator position on window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            initializeTabIndicator();
        }, 150);
    });
}

// Helper function to get current time range value
function getTimeRangeValue() {
    const activeItem = document.querySelector('.dropdown-item.active');
    return activeItem ? activeItem.getAttribute('data-value') : '24h';
}

function initCustomDropdown() {
    const dropdown = document.getElementById('timeRangeDropdown');
    const trigger = document.getElementById('timeRangeTrigger');
    const menu = document.getElementById('timeRangeMenu');
    const selected = document.getElementById('timeRangeSelected');
    const items = menu.querySelectorAll('.dropdown-item');

    if (!dropdown || !trigger || !menu || !selected) return;

    // Toggle dropdown
    trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = trigger.getAttribute('aria-expanded') === 'true';
        trigger.setAttribute('aria-expanded', !isOpen);
        menu.classList.toggle('show', !isOpen);

        // Update icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    });

    // Handle item selection
    items.forEach(item => {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            const value = item.getAttribute('data-value');
            const text = item.querySelector('span').textContent;

            // Update selected text
            selected.textContent = text;

            // Update active state
            items.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Close dropdown
            trigger.setAttribute('aria-expanded', 'false');
            menu.classList.remove('show');

            // Trigger change event
            const event = new CustomEvent('timeRangeChange', { detail: { value, text } });
            document.dispatchEvent(event);

            // Reload metrics
            loadAllMetrics();

            // Update icons
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target)) {
            trigger.setAttribute('aria-expanded', 'false');
            menu.classList.remove('show');
        }
    });

    // Keyboard navigation
    trigger.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            trigger.click();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (!menu.classList.contains('show')) {
                trigger.click();
            }
            const firstItem = items[0];
            if (firstItem) firstItem.focus();
        }
    });

    items.forEach((item, index) => {
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                item.click();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const nextItem = items[index + 1];
                if (nextItem) {
                    nextItem.focus();
                } else {
                    items[0].focus();
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prevItem = items[index - 1];
                if (prevItem) {
                    prevItem.focus();
                } else {
                    items[items.length - 1].focus();
                }
            } else if (e.key === 'Escape') {
                e.preventDefault();
                trigger.setAttribute('aria-expanded', 'false');
                menu.classList.remove('show');
                trigger.focus();
            }
        });

        // Make items focusable
        item.setAttribute('tabindex', '0');
    });

    // Set initial value (24h is default and already has active class in HTML)
    // Store current value for easy access
    window.currentTimeRange = getTimeRangeValue();

    // Listen for timeRange changes to update stored value
    document.addEventListener('timeRangeChange', (e) => {
        window.currentTimeRange = e.detail.value;
    });
}

// ===== TAB SWITCHING =====
function initializeTabIndicator() {
    const indicator = document.getElementById('navTabIndicator');
    const activeTab = document.querySelector('.nav-tab.active');
    const nav = document.querySelector('.dashboard-nav');

    if (indicator && activeTab && nav) {
        const navRect = nav.getBoundingClientRect();
        const tabRect = activeTab.getBoundingClientRect();

        const left = tabRect.left - navRect.left;
        const width = tabRect.width;

        indicator.style.left = `${left}px`;
        indicator.style.width = `${width}px`;
    }
}

function switchTab(tabName) {
    const tabs = document.querySelectorAll('.nav-tab');
    const contents = document.querySelectorAll('.tab-content');
    const indicator = document.getElementById('navTabIndicator');
    const nav = document.querySelector('.dashboard-nav');

    tabs.forEach(t => t.classList.remove('active'));
    contents.forEach(c => c.classList.remove('active'));

    const activeTab = Array.from(tabs).find(t => t.getAttribute('data-tab') === tabName);
    const activeContent = document.getElementById(tabName);

    if (activeTab) activeTab.classList.add('active');
    if (activeContent) activeContent.classList.add('active');

    // Animate the sliding indicator
    if (indicator && activeTab && nav) {
        const navRect = nav.getBoundingClientRect();
        const tabRect = activeTab.getBoundingClientRect();

        const left = tabRect.left - navRect.left;
        const width = tabRect.width;

        indicator.style.left = `${left}px`;
        indicator.style.width = `${width}px`;
    }

    currentTab = tabName;

    // Save the current tab to localStorage so it persists on refresh
    try {
        localStorage.setItem('dashboardActiveTab', tabName);
    } catch (e) {
        console.warn('Could not save tab to localStorage:', e);
    }

    // Load data and render charts for the active tab
    if (tabName === 'overview') {
        renderOverviewCharts();
    } else if (tabName === 'test-results') {
        loadTestResults();
    } else if (tabName === 'performance') {
        renderLatencyDistributionChart();
        renderResourceUtilizationMetrics();
    }

    // Re-initialize icons after tab switch
    if (typeof lucide !== 'undefined') {
        setTimeout(() => lucide.createIcons(), 100);
    }

    // Ultra-smooth scroll to top when switching tabs (optimized for 120 FPS)
    if (lenis) {
        lenis.scrollTo(0, {
            duration: targetFPS >= 120 ? 0.6 : 0.8, // Faster on high refresh displays
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // Ultra-smooth easing
            immediate: false,
        });
    } else {
        requestAnimationFrame(() => {
            window.scrollTo({ top: 0, behavior: 'auto' });
        });
    }
}

// ===== WEBSOCKET =====
let wsReconnectTimeout = null;

function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }

    if (wsReconnectTimeout) {
        clearTimeout(wsReconnectTimeout);
        wsReconnectTimeout = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        // WebSocket connected
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'metrics_update') {
                updateMetricsFromWebSocket(data.data);
            }
        } catch (e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };

    ws.onclose = () => {
        wsReconnectTimeout = setTimeout(() => {
            initWebSocket();
        }, 5000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// ===== API CALLS =====
let ragTargets = null; // Cache for RAG targets

async function loadRAGTargets() {
    try {
        const response = await fetch(`/api/rag/targets?_t=${Date.now()}`);
        ragTargets = await response.json();
        console.log('Loaded RAG targets:', ragTargets);

        // Update target labels in UI
        updateRAGTargetLabels();

        return ragTargets;
    } catch (error) {
        console.warn('Failed to load RAG targets, using defaults:', error);
        // Return default targets
        ragTargets = {
            retrieval_recall_5: 85.0,
            retrieval_precision_5: 85.0,
            context_relevance: 80.0,
            context_coverage: 80.0,
            context_intrusion: 5.0,
            gold_context_match: 85.0,
            reranker_score: 0.8,
        };
        updateRAGTargetLabels();
        return ragTargets;
    }
}

function updateRAGTargetLabels() {
    if (!ragTargets) return;

    // Update Recall@5 target
    const recallTargetEl = document.getElementById('ragRecallTarget');
    if (recallTargetEl) {
        recallTargetEl.textContent = `Target: > ${Math.round(ragTargets.retrieval_recall_5)}%`;
    }

    // Update Precision@5 target
    const precisionTargetEl = document.getElementById('ragPrecisionTarget');
    if (precisionTargetEl) {
        precisionTargetEl.textContent = `Target: > ${Math.round(ragTargets.retrieval_precision_5)}%`;
    }

    // Update Context Intrusion target (lower is better)
    const intrusionTargetEl = document.getElementById('ragIntrusionTarget');
    if (intrusionTargetEl) {
        intrusionTargetEl.textContent = `Target: < ${ragTargets.context_intrusion.toFixed(1)}%`;
    }

    // Update Reranker Score target
    const rerankerTargetEl = document.getElementById('ragRerankerTarget');
    if (rerankerTargetEl) {
        rerankerTargetEl.textContent = `Target: > ${ragTargets.reranker_score.toFixed(2)}`;
    }
}

async function loadAllMetrics() {
    try {
        // Load RAG targets first (if not already loaded)
        if (!ragTargets) {
            ragTargets = await loadRAGTargets();
        }

        // Load statistics
        const statsResponse = await fetch(`/api/statistics?_t=${Date.now()}`);
        const stats = await statsResponse.json();

        // Calculate and populate metrics
        calculateMetricsFromStats(stats);

        // Update UI
        updateOverviewKPIs(stats);
        updateCriticalMetrics(stats);
        updateBaseModelMetrics();
        updateRAGMetrics();
        updateSafetyMetrics();
        updatePerformanceMetrics();
        updateReliabilityMetrics();
        updateAgentMetrics();
        updateSecurityMetrics(stats);
        updateFooterStats(stats);

        // Render charts
        if (currentTab === 'overview') {
            setTimeout(() => renderOverviewCharts(), 100);
        }

    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

function calculateMetricsFromStats(stats) {
    // Calculate overall health score
    const totalTests = stats.total_tests || 0;
    const passed = stats.passed || 0;
    const passRate = totalTests > 0 ? (passed / totalTests * 100) : 0;

    // Helper function to get metric value or return null (no mock data)
    const getMetric = (metrics, key, defaultValue = null) => {
        const value = metrics[key];
        return value !== undefined && value !== null ? value : defaultValue;
    };

    // Base Model Metrics (derived from validation metrics)
    const validationMetrics = stats.validation_metrics || {};

    // Helper to convert 0-1 range to percentage if needed
    // Note: Backend converts 0-1 metrics to 0-100 in data_store.py, but we handle both cases
    const convertToPercentage = (value) => {
        if (value === null || value === undefined) return null;
        // If value is between 0-1 (and not already a whole number), convert to percentage (0-100)
        if (value >= 0 && value <= 1 && value < 1 && !Number.isInteger(value)) {
            return value * 100;
        }
        // If value is already a percentage (0-100), return as-is
        return value;
    };

    // Helper to convert decimal scores (0-1) to percentage for display
    // BLEU, ROUGE-L, BERTScore are typically 0-1 range and should be shown as percentages
    const convertDecimalToPercentage = (value) => {
        if (value === null || value === undefined) return null;
        // If value is between 0-1, convert to percentage
        if (value >= 0 && value <= 1) {
            return value * 100;
        }
        // If already a percentage, return as-is
        return value;
    };

    metricsData.baseModel = {
        accuracy: convertToPercentage(getMetric(validationMetrics, 'accuracy')),
        normalizedSimilarityScore: convertToPercentage(getMetric(validationMetrics, 'normalized_similarity_score')),
        exactMatch: convertToPercentage(getMetric(validationMetrics, 'exact_match')),
        f1Score: convertToPercentage(getMetric(validationMetrics, 'f1_score')),
        bleu: convertDecimalToPercentage(getMetric(validationMetrics, 'bleu')), // Convert 0-1 to 0-100 for display
        rougeL: convertDecimalToPercentage(getMetric(validationMetrics, 'rouge_l')), // Convert 0-1 to 0-100 for display
        bertScore: convertDecimalToPercentage(getMetric(validationMetrics, 'bert_score')), // Convert 0-1 to 0-100 for display
        cotValidity: convertToPercentage(getMetric(validationMetrics, 'cot_validity')),
        stepCorrectness: convertToPercentage(getMetric(validationMetrics, 'step_correctness')),
        logicConsistency: convertToPercentage(getMetric(validationMetrics, 'logic_consistency')),
        hallucinationRate: getMetric(validationMetrics, 'hallucination_rate'), // Already in percentage
        similarityProxyFactualConsistency: getMetric(validationMetrics, 'similarity_proxy_factual_consistency'), // Already in percentage
        similarityProxyTruthfulness: getMetric(validationMetrics, 'similarity_proxy_truthfulness'), // Already in percentage
        citationAccuracy: getMetric(validationMetrics, 'citation_accuracy'), // Already in percentage
        similarityProxySourceGrounding: getMetric(validationMetrics, 'similarity_proxy_source_grounding') // Already in percentage
    };

    // RAG Metrics
    metricsData.rag = {
        retrievalRecall5: getMetric(validationMetrics, 'retrieval_recall_5'),
        retrievalPrecision5: getMetric(validationMetrics, 'retrieval_precision_5'),
        contextRelevance: getMetric(validationMetrics, 'context_relevance'),
        contextCoverage: getMetric(validationMetrics, 'context_coverage'),
        contextIntrusion: getMetric(validationMetrics, 'context_intrusion'),
        goldContextMatch: getMetric(validationMetrics, 'gold_context_match'),
        rerankerScore: getMetric(validationMetrics, 'reranker_score')
    };

    // Safety Metrics
    // Note: 0 is a VALID value (meaning no toxicity/bias detected), not missing data
    // The metrics calculator returns 0 when no issues are detected, which is correct
    // For safety metrics, 0% means "no issues found" (good), not "no data"
    metricsData.safety = {
        toxicityScore: getMetric(validationMetrics, 'toxicity_score'),
        biasScore: getMetric(validationMetrics, 'bias_score'),
        promptInjection: getMetric(validationMetrics, 'prompt_injection'),
        refusalRate: getMetric(validationMetrics, 'refusal_rate'),
        complianceScore: getMetric(validationMetrics, 'compliance_score'),
        dataLeakage: getMetric(validationMetrics, 'data_leakage'),
        harmfulnessScore: getMetric(validationMetrics, 'harmfulness_score'),
        ethicalViolation: getMetric(validationMetrics, 'ethical_violation'),
        piiLeakage: getMetric(validationMetrics, 'pii_leakage')
    };

    // Performance Metrics - use calculated metrics
    const avgDuration = stats.avg_duration || null;
    const systemMetrics = stats.system_metrics || {};

    metricsData.performance = {
        tokenLatency: getMetric(validationMetrics, 'token_latency'),
        e2eLatency: avgDuration ? Math.round(avgDuration * 1000) : getMetric(validationMetrics, 'e2e_latency'),
        throughput: getMetric(validationMetrics, 'throughput'),
        ttft: getMetric(validationMetrics, 'ttft'),
        gpuUtil: systemMetrics.gpu_util !== undefined ? systemMetrics.gpu_util : (systemMetrics.gpuUtil !== undefined ? systemMetrics.gpuUtil : null),
        memFootprint: systemMetrics.mem_footprint !== undefined ? systemMetrics.mem_footprint : (systemMetrics.memFootprint !== undefined ? systemMetrics.memFootprint : null),
        cpuUtil: systemMetrics.cpu_util !== undefined ? systemMetrics.cpu_util : (systemMetrics.cpuUtil !== undefined ? systemMetrics.cpuUtil : null),
        timeoutRate: stats.timeout_rate || null
    };

    // Reliability Metrics - use calculated metrics
    metricsData.reliability = {
        determinismScore: getMetric(validationMetrics, 'determinism_score'),
        outputStability: getMetric(validationMetrics, 'output_stability'),
        crashRate: getMetric(validationMetrics, 'crash_rate'), // Try to get from metrics
        retryRate: getMetric(validationMetrics, 'retry_rate'), // Try to get from metrics
        completionSuccess: passRate, // Real data from stats - will be rounded in updateElement
        schemaCompliance: getMetric(validationMetrics, 'schema_compliance'),
        outputValidity: getMetric(validationMetrics, 'output_validity'),
        jsonValidity: getMetric(validationMetrics, 'json_validity'), // Try to get from metrics
        typeCorrectness: getMetric(validationMetrics, 'type_correctness') // Try to get from metrics
    };

    // Agent Metrics - use calculated metrics
    metricsData.agent = {
        taskCompletion: getMetric(validationMetrics, 'task_completion'),
        stepEfficiency: getMetric(validationMetrics, 'step_efficiency'),
        errorRecovery: getMetric(validationMetrics, 'error_recovery'),
        toolUsageAccuracy: getMetric(validationMetrics, 'tool_usage_accuracy'),
        planningCoherence: getMetric(validationMetrics, 'planning_coherence'),
        actionHallucination: getMetric(validationMetrics, 'action_hallucination'),
        goalDrift: getMetric(validationMetrics, 'goal_drift')
    };

    // Security Metrics - use calculated metrics
    metricsData.security = {
        injectionAttackSuccess: getMetric(validationMetrics, 'injection_attack_success'),
        adversarialVulnerability: getMetric(validationMetrics, 'adversarial_vulnerability'),
        dataExfiltration: getMetric(validationMetrics, 'data_exfiltration'),
        modelEvasion: getMetric(validationMetrics, 'model_evasion'),
        extractionRisk: getMetric(validationMetrics, 'extraction_risk')
    };
}

// Shared function to calculate Safety Score consistently
// Safety includes: Content Safety, Data Protection, and Attack Resistance (prompt injection)
// NOTE: Security metrics (injectionAttackSuccess, etc.) are separate and should NOT be included here
function calculateSafetyScore(roundToInteger = false) {
    const safetyMetrics = [
        // Content Safety (inverted - lower is better)
        metricsData.safety.toxicityScore !== null ? 100 - metricsData.safety.toxicityScore : null,
        metricsData.safety.biasScore !== null ? 100 - metricsData.safety.biasScore : null,
        metricsData.safety.harmfulnessScore !== null ? 100 - metricsData.safety.harmfulnessScore : null,
        metricsData.safety.ethicalViolation !== null ? 100 - metricsData.safety.ethicalViolation : null,
        // Data Protection
        metricsData.safety.complianceScore, // Direct - higher is better
        metricsData.safety.dataLeakage !== null ? 100 - metricsData.safety.dataLeakage : null,
        metricsData.safety.piiLeakage !== null ? 100 - metricsData.safety.piiLeakage : null,
        // Attack Resistance (prompt injection is safety-related, not security)
        metricsData.safety.promptInjection !== null ? 100 - metricsData.safety.promptInjection : null
    ].filter(v => v !== null && v !== undefined);

    if (safetyMetrics.length === 0) return null;

    const score = safetyMetrics.reduce((a, b) => a + b, 0) / safetyMetrics.length;
    return roundToInteger ? Math.round(score) : parseFloat(score.toFixed(1));
}

function updateOverviewKPIs(stats) {
    const totalTests = stats.total_tests || 0;
    const passed = stats.passed || 0;
    const passRate = totalTests > 0 ? (passed / totalTests * 100) : 0;
    const avgDuration = stats.avg_duration || null;
    const trends = stats.trends || {};

    // Overall Health Score - calculate from available metrics
    const healthComponents = [];
    if (passRate > 0) healthComponents.push(passRate);
    if (metricsData.reliability.completionSuccess !== null) {
        healthComponents.push(metricsData.reliability.completionSuccess);
    }
    if (metricsData.safety.complianceScore !== null) {
        healthComponents.push(metricsData.safety.complianceScore);
    }
    const healthScore = healthComponents.length > 0
        ? (healthComponents.reduce((a, b) => a + b, 0) / healthComponents.length).toFixed(1)
        : null;
    updateElement('overallHealth', healthScore, 'percentage');

    // Update health trend
    const healthTrend = trends.health_trend || 0;
    updateTrend('overallHealth', healthTrend);

    // Avg Response Time
    updateElement('avgResponseTime', avgDuration ? `${avgDuration.toFixed(2)}s` : null);

    // Update duration trend (inverse - higher duration is worse)
    const durationTrend = trends.duration_trend || 0;
    updateTrend('avgResponseTime', -durationTrend); // Negative because higher duration is worse

    // Safety Score - use shared calculation function
    const safetyScore = calculateSafetyScore(false); // false = use one decimal place
    updateElement('safetyScore', safetyScore !== null ? safetyScore.toFixed(1) : null, 'percentage');

    // Update safety trend
    const safetyTrend = trends.safety_trend || 0;
    updateTrend('safetyScore', safetyTrend);

    // User Satisfaction (derived from pass rate)
    const satisfaction = passRate > 0 ? (3.5 + (passRate / 100) * 1.5).toFixed(1) : null;
    updateElement('userSatisfaction', satisfaction ? `${satisfaction}/5` : null);

    // Update satisfaction trend
    const satisfactionTrend = trends.satisfaction_trend || 0;
    updateTrend('userSatisfaction', satisfactionTrend);
}

function updateTrend(elementId, trendValue) {
    // Find the trend element for this KPI card
    const kpiCard = document.getElementById(elementId)?.closest('.kpi-card');
    if (!kpiCard) return;

    const trendElement = kpiCard.querySelector('.kpi-trend');
    if (!trendElement) return;

    // Determine if positive or negative based on context
    // For health, safety, satisfaction: positive trend is good (up arrow)
    // For response time: positive trend is bad (down arrow)
    const isPositive = elementId === 'avgResponseTime' ? trendValue < 0 : trendValue > 0;
    const absValue = Math.abs(trendValue);

    // Update class and content
    trendElement.className = `kpi-trend ${isPositive ? 'positive' : 'negative'}`;
    trendElement.textContent = `${isPositive ? '↑' : '↓'} ${absValue.toFixed(1)}%`;
}

function updateCriticalMetrics(stats = null) {
    const trends = stats?.trends || {};

    const criticalMetrics = [
        { name: 'Hallucination Rate', value: metricsData.baseModel.hallucinationRate, target: 5, unit: '%', inverse: true },
        { name: 'Source Grounding (Similarity Proxy)', value: metricsData.baseModel.similarityProxySourceGrounding, target: 85, unit: '%' },
        { name: 'Factual Consistency (Similarity Proxy)', value: metricsData.baseModel.similarityProxyFactualConsistency, target: 90, unit: '%' },
        { name: 'Response Validity', value: metricsData.reliability.outputValidity, target: 95, unit: '%' },
        { name: 'Schema Compliance', value: metricsData.reliability.schemaCompliance, target: 95, unit: '%' },
        { name: 'Truthfulness (Similarity Proxy)', value: metricsData.baseModel.similarityProxyTruthfulness, target: 90, unit: '%' },
        { name: 'Token Latency (TTFT)', value: metricsData.performance.ttft, target: 300, unit: 'ms', inverse: true },
        { name: 'E2E Latency', value: metricsData.performance.e2eLatency, target: 2000, unit: 'ms', inverse: true },
        { name: 'Retrieval Precision@5', value: metricsData.rag.retrievalPrecision5, target: ragTargets?.retrieval_precision_5 || 85, unit: '%' },
        { name: 'Retrieval Recall@5', value: metricsData.rag.retrievalRecall5, target: ragTargets?.retrieval_recall_5 || 85, unit: '%' },
        { name: 'Context Coverage', value: metricsData.rag.contextCoverage, target: ragTargets?.context_coverage || 80, unit: '%' },
        { name: 'Reasoning Correctness', value: metricsData.baseModel.stepCorrectness, target: 90, unit: '%' },
        { name: 'Output Stability', value: metricsData.reliability.outputStability, target: 90, unit: '%' },
        { name: 'Safety Violations', value: metricsData.safety.harmfulnessScore, target: 2, unit: '%', inverse: true },
        { name: 'Injection Vulnerability', value: metricsData.security.injectionAttackSuccess, target: 2, unit: '%', inverse: true }
    ];

    const grid = document.getElementById('criticalMetricsGrid');
    if (!grid) return;

    let excellentCount = 0;
    let goodCount = 0;
    let watchCount = 0;
    let totalValue = 0;
    let validMetricsCount = 0;

    grid.innerHTML = criticalMetrics.map((metric, idx) => {
        // Handle metrics with no data
        if (metric.value === null || metric.value === undefined) {
            return `
                <div class="critical-metric-card na">
                    <div class="critical-metric-header">
                        <div class="flex-1">
                            <p class="critical-metric-name">${escapeHtml(metric.name)}</p>
                        </div>
                        <div class="critical-metric-status-icon">
                            <span class="critical-metric-status-icon-text" style="color: #64748b;">-</span>
                        </div>
                    </div>
                    <div class="critical-metric-value-container">
                        <span class="critical-metric-value na">N/A</span>
                    </div>
                    <div class="critical-metric-progress-bar">
                        <div class="critical-metric-progress-bar-bg">
                            <div class="critical-metric-progress-bar-fill" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="critical-metric-target-row">
                        <span class="critical-metric-target">Target: ${metric.target}${metric.unit}</span>
                        <span class="critical-metric-status-text" style="color: #64748b;">-</span>
                    </div>
                    <div class="critical-metric-rank-badge">
                        <span class="critical-metric-rank-number">${idx + 1}</span>
                    </div>
                </div>
            `;
        }

        // Calculate status based on ratio
        let ratio;
        if (metric.inverse) {
            // For inverse metrics (lower is better), ratio = target / value
            // Handle division by zero: if value is 0 or very small, cap at 2.0 (200%)
            if (metric.value === 0 || metric.value === null || metric.value < 0.01) {
                ratio = 2.0; // Cap at 200% for perfect scores (0 violations)
            } else {
                ratio = metric.target / metric.value;
                // Cap ratio at 2.0 (200%) to prevent Infinity
                ratio = Math.min(ratio, 2.0);
            }
        } else {
            // For normal metrics (higher is better), ratio = value / target
            ratio = metric.value / metric.target;
            // Cap ratio at 2.0 (200%) to prevent extremely high values
            ratio = Math.min(ratio, 2.0);
        }

        let status = '';
        let statusIcon = '';
        let statusText = '';

        if (ratio >= 1) {
            status = 'excellent';
            statusIcon = '✓';
            statusText = '✓ Met';
            excellentCount++;
        } else if (ratio >= 0.9) {
            status = 'good';
            statusIcon = '◉';
            statusText = `${(ratio * 100).toFixed(0)}%`;
            goodCount++;
        } else {
            status = 'warning';
            statusIcon = '⚠';
            statusText = `${(ratio * 100).toFixed(0)}%`;
            watchCount++;
        }

        // Calculate progress for display (capped at 100%)
        const progressWidth = Math.min(ratio * 100, 100);

        // Add to system health calculation (cap at 100% for health score)
        const healthContribution = Math.min(ratio * 100, 100);
        totalValue += healthContribution;
        validMetricsCount++;

        return `
            <div class="critical-metric-card ${status}">
                <div class="critical-metric-header">
                    <div class="flex-1">
                        <p class="critical-metric-name">${escapeHtml(metric.name)}</p>
                    </div>
                    <div class="critical-metric-status-icon ${status}">
                        <span class="critical-metric-status-icon-text ${status}">${statusIcon}</span>
                    </div>
                </div>
                <div class="critical-metric-value-container">
                    <span class="critical-metric-value ${status}">${metric.value}</span>
                    <span class="critical-metric-unit">${metric.unit}</span>
                </div>
                <div class="critical-metric-progress-bar">
                    <div class="critical-metric-progress-bar-bg">
                        <div class="critical-metric-progress-bar-fill ${status}" style="width: ${progressWidth}%"></div>
                    </div>
                </div>
                <div class="critical-metric-target-row">
                    <span class="critical-metric-target">Target: ${metric.target}${metric.unit}</span>
                    <span class="critical-metric-status-text ${status}">${statusText}</span>
                </div>
                <div class="critical-metric-rank-badge">
                    <span class="critical-metric-rank-number">${idx + 1}</span>
                </div>
            </div>
        `;
    }).join('');

    // Update summary stats
    updateElement('criticalMetricsExcellent', excellentCount);
    updateElement('criticalMetricsGood', goodCount);
    updateElement('criticalMetricsWatch', watchCount);

    // Calculate and update system health
    const systemHealth = validMetricsCount > 0 ? (totalValue / validMetricsCount).toFixed(1) : null;
    updateElement('criticalMetricsSystemHealth', systemHealth ? `${systemHealth}%` : 'N/A');

    // Update metrics count
    const totalMetrics = criticalMetrics.length;
    const validCount = validMetricsCount;
    updateElement('criticalMetricsCount', `${validCount}/${totalMetrics}`);

    // Calculate trend (use health trend if available, otherwise 0)
    // Use the health_trend from stats if available, which compares current vs previous period
    const trendValue = trends.health_trend !== undefined ? trends.health_trend : 0;
    const trendDisplay = trendValue >= 0 ? `+${trendValue.toFixed(1)}` : trendValue.toFixed(1);
    updateElement('criticalMetricsTrend', `${trendDisplay}%`);

    // Re-initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function updateBaseModelMetrics() {
    updateElement('baseAccuracy', metricsData.baseModel.accuracy, 'percentage');
    updateElement('baseTopK', metricsData.baseModel.normalizedSimilarityScore, 'percentage');
    updateElement('baseExactMatch', metricsData.baseModel.exactMatch, 'percentage');
    updateElement('baseF1', metricsData.baseModel.f1Score, 'percentage');
    // BLEU, ROUGE-L, BERTScore are now converted to percentages for consistency
    updateElement('baseBLEU', metricsData.baseModel.bleu, 'percentage');
    updateElement('baseROUGE', metricsData.baseModel.rougeL, 'percentage');
    updateElement('baseBERT', metricsData.baseModel.bertScore, 'percentage');
    updateElement('baseCoT', metricsData.baseModel.cotValidity, 'percentage');
    updateElement('baseStepCorrect', metricsData.baseModel.stepCorrectness, 'percentage');
    updateElement('baseLogic', metricsData.baseModel.logicConsistency, 'percentage');
    updateElement('hallucinationRate', metricsData.baseModel.hallucinationRate, 'percentage');
    updateElement('factualConsistency', metricsData.baseModel.similarityProxyFactualConsistency, 'percentage');
    updateElement('truthfulness', metricsData.baseModel.similarityProxyTruthfulness, 'percentage');
    updateElement('citationAccuracy', metricsData.baseModel.citationAccuracy, 'percentage');
    updateElement('sourceGrounding', metricsData.baseModel.similarityProxySourceGrounding, 'percentage');
}

function updateRAGMetrics() {
    // Update main metrics
    updateElement('ragRecall', metricsData.rag.retrievalRecall5, 'percentage');
    updateElement('ragPrecision', metricsData.rag.retrievalPrecision5, 'percentage');
    updateElement('ragRelevance', metricsData.rag.contextRelevance, 'percentage');
    updateElement('ragCoverage', metricsData.rag.contextCoverage, 'percentage');
    updateElement('ragGoldMatch', metricsData.rag.goldContextMatch, 'percentage');
    updateElement('ragIntrusion', metricsData.rag.contextIntrusion, 'percentage');
    updateElement('ragReranker', metricsData.rag.rerankerScore, 'decimal');

    // Update progress bars with animations
    // Reset to 0% if no data, otherwise update with actual value
    if (metricsData.rag.retrievalRecall5 !== null && metricsData.rag.retrievalRecall5 !== undefined) {
        updateProgressBarWidth('ragRecallProgress', metricsData.rag.retrievalRecall5);
    } else {
        updateProgressBarWidth('ragRecallProgress', 0);
    }
    if (metricsData.rag.retrievalPrecision5 !== null && metricsData.rag.retrievalPrecision5 !== undefined) {
        updateProgressBarWidth('ragPrecisionProgress', metricsData.rag.retrievalPrecision5);
    } else {
        updateProgressBarWidth('ragPrecisionProgress', 0);
    }
    if (metricsData.rag.contextRelevance !== null && metricsData.rag.contextRelevance !== undefined) {
        updateProgressBarWidth('ragRelevanceProgress', metricsData.rag.contextRelevance);
    } else {
        updateProgressBarWidth('ragRelevanceProgress', 0);
    }
    if (metricsData.rag.contextCoverage !== null && metricsData.rag.contextCoverage !== undefined) {
        updateProgressBarWidth('ragCoverageProgress', metricsData.rag.contextCoverage);
    } else {
        updateProgressBarWidth('ragCoverageProgress', 0);
    }
    if (metricsData.rag.goldContextMatch !== null && metricsData.rag.goldContextMatch !== undefined) {
        updateProgressBarWidth('ragGoldMatchProgress', metricsData.rag.goldContextMatch);
    } else {
        updateProgressBarWidth('ragGoldMatchProgress', 0);
    }

        // Update context intrusion status (use calibrated target)
        if (metricsData.rag.contextIntrusion !== null) {
            const intrusionEl = document.getElementById('ragIntrusionStatus');
            if (intrusionEl) {
                const indicator = intrusionEl.querySelector('.status-indicator');
                const text = intrusionEl.querySelector('span:last-child');
                const target = ragTargets?.context_intrusion || 5.0;
                if (metricsData.rag.contextIntrusion < target) {
                    indicator.className = 'status-indicator status-good';
                    text.textContent = 'Within Target';
                } else if (metricsData.rag.contextIntrusion < target * 2) {
                    indicator.className = 'status-indicator status-warning';
                    text.textContent = 'Needs Attention';
                } else {
                    indicator.className = 'status-indicator status-danger';
                    text.textContent = 'High Intrusion';
                }
            }
        }

        // Update reranker status (use calibrated target)
        if (metricsData.rag.rerankerScore !== null) {
            const rerankerEl = document.getElementById('ragRerankerStatus');
            if (rerankerEl) {
                const indicator = rerankerEl.querySelector('.status-indicator');
                const text = rerankerEl.querySelector('span:last-child');
                const target = ragTargets?.reranker_score || 0.8;
                if (metricsData.rag.rerankerScore >= target) {
                    indicator.className = 'status-indicator status-good';
                    text.textContent = 'Above Target';
                } else if (metricsData.rag.rerankerScore >= target * 0.75) {
                    indicator.className = 'status-indicator status-warning';
                    text.textContent = 'Below Target';
                } else {
                    indicator.className = 'status-indicator status-danger';
                    text.textContent = 'Poor Score';
                }
            }
        }

    // Calculate overall RAG health if we have data
    const ragMetrics = [
        metricsData.rag.retrievalRecall5,
        metricsData.rag.contextRelevance,
        metricsData.rag.contextCoverage,
        metricsData.rag.goldContextMatch
    ].filter(v => v !== null);
    const ragHealth = ragMetrics.length > 0
        ? (ragMetrics.reduce((a, b) => a + b, 0) / ragMetrics.length).toFixed(1)
        : null;
    updateElement('ragHealth', ragHealth, 'percentage');

    // Update health breakdown
    if (ragMetrics.length >= 2) {
        const retrievalAvg = metricsData.rag.retrievalRecall5 || 0;
        const qualityAvg = [
            metricsData.rag.contextRelevance,
            metricsData.rag.contextCoverage,
            metricsData.rag.goldContextMatch
        ].filter(v => v !== null);
        const qualityScore = qualityAvg.length > 0
            ? (qualityAvg.reduce((a, b) => a + b, 0) / qualityAvg.length).toFixed(1)
            : 0;

        updateElement('ragHealthRetrieval', retrievalAvg, 'percentage');
        updateElement('ragHealthQuality', qualityScore, 'percentage');
    }

    // Re-initialize Lucide icons after DOM updates
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function updateRAGBadge(elementId, value, threshold) {
    const badgeEl = document.getElementById(elementId);
    if (!badgeEl) return;

    if (value >= threshold * 1.2) {
        badgeEl.textContent = 'Excellent';
        badgeEl.style.background = 'rgba(16, 185, 129, 0.2)';
        badgeEl.style.color = '#10b981';
        badgeEl.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else if (value >= threshold) {
        badgeEl.textContent = 'Good';
        badgeEl.style.background = 'rgba(16, 185, 129, 0.2)';
        badgeEl.style.color = '#10b981';
        badgeEl.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else if (value >= threshold * 0.8) {
        badgeEl.textContent = 'Fair';
        badgeEl.style.background = 'rgba(245, 158, 11, 0.2)';
        badgeEl.style.color = '#f59e0b';
        badgeEl.style.borderColor = 'rgba(245, 158, 11, 0.3)';
    } else {
        badgeEl.textContent = 'Poor';
        badgeEl.style.background = 'rgba(239, 68, 68, 0.2)';
        badgeEl.style.color = '#ef4444';
        badgeEl.style.borderColor = 'rgba(239, 68, 68, 0.3)';
    }
}

function updateSafetyMetrics() {
    updateElement('safetyToxicity', metricsData.safety.toxicityScore, 'percentage');
    updateElement('safetyBias', metricsData.safety.biasScore, 'percentage');
    updateElement('safetyHarm', metricsData.safety.harmfulnessScore, 'percentage');
    updateElement('safetyEthical', metricsData.safety.ethicalViolation, 'percentage');
    updateElement('safetyDataLeak', metricsData.safety.dataLeakage, 'percentage');
    updateElement('safetyPII', metricsData.safety.piiLeakage, 'percentage');
    updateElement('safetyCompliance', metricsData.safety.complianceScore, 'percentage');
    updateElement('safetyInjection', metricsData.safety.promptInjection, 'percentage');
    updateElement('safetyRefusal', metricsData.safety.refusalRate, 'percentage');

    // Use shared Safety Score calculation function for consistency
    const safetyScore = calculateSafetyScore(false); // false = use one decimal place
    updateElement('overallSafetyScore', safetyScore !== null ? safetyScore.toFixed(1) : null, 'percentage');
}

function updatePerformanceMetrics() {
    // Format token latency (round to 2 decimals if > 1, otherwise show more precision)
    const tokenLatency = metricsData.performance.tokenLatency;
    const tokenLatencyFormatted = tokenLatency !== null
        ? (tokenLatency > 1 ? tokenLatency.toFixed(2) : tokenLatency.toFixed(4))
        : null;
    updateElement('perfTokenLatency', tokenLatencyFormatted);

    // Format TTFT (round to nearest integer for ms)
    const ttft = metricsData.performance.ttft;
    const ttftFormatted = ttft !== null ? Math.round(ttft) : null;
    updateElement('perfTTFT', ttftFormatted);

    // Format E2E latency (round to nearest integer for ms)
    const e2eLatency = metricsData.performance.e2eLatency;
    const e2eLatencyFormatted = e2eLatency !== null ? Math.round(e2eLatency) : null;
    updateElement('perfE2E', e2eLatencyFormatted);

    // Format throughput (round to nearest integer for tokens/sec)
    const throughput = metricsData.performance.throughput;
    const throughputFormatted = throughput !== null
        ? Math.round(throughput).toLocaleString()
        : null;
    updateElement('perfThroughput', throughputFormatted);

    // Render resource utilization metrics
    renderResourceUtilizationMetrics();
}

function renderResourceUtilizationMetrics() {
    const container = document.getElementById('resourceUtilizationChart');
    if (!container) return;

    try {
        // Get resource utilization data
        const safeParse = (val) => {
            if (val === null || val === undefined) return null;
            const parsed = parseFloat(val);
            return isNaN(parsed) ? null : parsed;
        };

        const gpuUtil = safeParse(metricsData.performance.gpuUtil);
        const cpuUtil = safeParse(metricsData.performance.cpuUtil);
        const memFootprint = safeParse(metricsData.performance.memFootprint);
        const timeoutRate = safeParse(metricsData.performance.timeoutRate);

        // Convert values to percentages
        const maxMemoryGB = 100;
        const gpuValue = gpuUtil !== null ? Math.min(Math.max(gpuUtil, 0), 100) : 0;
        const cpuValue = cpuUtil !== null ? Math.min(Math.max(cpuUtil, 0), 100) : 0;
        const memValue = memFootprint !== null
            ? Math.min(Math.max((memFootprint / maxMemoryGB) * 100, 0), 100)
            : 0;
        const timeoutValue = timeoutRate !== null ? Math.min(Math.max(timeoutRate, 0), 100) : 0;

        // Metrics array matching the React component structure
        const metrics = [
            { label: 'GPU', value: gpuValue, color: '#3b82f6', rawValue: gpuUtil, formatted: gpuUtil !== null ? `${gpuUtil.toFixed(1)}%` : 'N/A' },
            { label: 'CPU', value: cpuValue, color: '#10b981', rawValue: cpuUtil, formatted: cpuUtil !== null ? `${cpuUtil.toFixed(1)}%` : 'N/A' },
            { label: 'Memory', value: memValue, color: '#8b5cf6', rawValue: memFootprint, formatted: memFootprint !== null ? `${memFootprint.toFixed(1)} GB` : 'N/A' },
            { label: 'Timeout Rate', value: timeoutValue, color: '#06b6d4', rawValue: timeoutRate, formatted: timeoutRate !== null ? `${timeoutRate.toFixed(1)}%` : 'N/A' }
        ];

        // If chart exists, update it instead of recreating
        if (charts.resourceUtilization && charts.resourceUtilization.update) {
            charts.resourceUtilization.update(metrics);
            return;
        }

        // Clear container
        container.innerHTML = '';
        container.style.cssText = `
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        `;

        // SVG Radial Tubes
        const svgContainer = document.createElement('div');
        svgContainer.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '600');
        svg.setAttribute('height', '410');
        svg.setAttribute('viewBox', '0 0 600 600');
        svg.style.cssText = 'transform: rotate(-90deg);';

        // SVG Definitions
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        metrics.forEach((metric, idx) => {
            // Linear Gradient
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.setAttribute('id', `gradient-${idx}`);
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '0%');
            gradient.setAttribute('y2', '100%');

            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', metric.color);
            stop1.setAttribute('stop-opacity', '0.3');

            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '50%');
            stop2.setAttribute('stop-color', metric.color);
            stop2.setAttribute('stop-opacity', '1');

            const stop3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop3.setAttribute('offset', '100%');
            stop3.setAttribute('stop-color', metric.color);
            stop3.setAttribute('stop-opacity', '0.4');

            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            gradient.appendChild(stop3);
            defs.appendChild(gradient);

            // Glow Filter
            const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
            filter.setAttribute('id', `glow-${idx}`);

            const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
            feGaussianBlur.setAttribute('stdDeviation', '5');
            feGaussianBlur.setAttribute('result', 'coloredBlur');

            const feMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
            for (let i = 0; i < 3; i++) {
                const feMergeNode = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
                if (i < 2) {
                    feMergeNode.setAttribute('in', 'coloredBlur');
                } else {
                    feMergeNode.setAttribute('in', 'SourceGraphic');
                }
                feMerge.appendChild(feMergeNode);
            }

            filter.appendChild(feGaussianBlur);
            filter.appendChild(feMerge);
            defs.appendChild(filter);
        });

        svg.appendChild(defs);

        // Chart parameters
        const centerX = 300;
        const centerY = 300;
        const baseRadius = 70;
        const tubeWidth = 35;
        const gap = 10;

        // Create circles for each metric
        metrics.forEach((metric, idx) => {
            const radius = baseRadius + (idx * (tubeWidth + gap));
            const circumference = 2 * Math.PI * radius;
            // Start at 0% (full offset) for animation
            const initialOffset = circumference;
            const finalOffset = circumference - (metric.value / 100) * circumference;

            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

            // Background track - dark
            const bgTrack = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            bgTrack.setAttribute('cx', centerX);
            bgTrack.setAttribute('cy', centerY);
            bgTrack.setAttribute('r', radius);
            bgTrack.setAttribute('fill', 'none');
            bgTrack.setAttribute('stroke', '#0f172a');
            bgTrack.setAttribute('stroke-width', tubeWidth);
            bgTrack.setAttribute('opacity', '0.6');
            g.appendChild(bgTrack);

            // Inner shadow for depth
            const innerShadow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            innerShadow.setAttribute('cx', centerX);
            innerShadow.setAttribute('cy', centerY);
            innerShadow.setAttribute('r', radius);
            innerShadow.setAttribute('fill', 'none');
            innerShadow.setAttribute('stroke', '#000000');
            innerShadow.setAttribute('stroke-width', tubeWidth);
            innerShadow.setAttribute('opacity', '0.3');
            g.appendChild(innerShadow);

            // Main progress tube with gradient
            const progressTube = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            progressTube.setAttribute('cx', centerX);
            progressTube.setAttribute('cy', centerY);
            progressTube.setAttribute('r', radius);
            progressTube.setAttribute('fill', 'none');
            progressTube.setAttribute('stroke', `url(#gradient-${idx})`);
            progressTube.setAttribute('stroke-width', tubeWidth);
            progressTube.setAttribute('stroke-dasharray', circumference);
            progressTube.setAttribute('stroke-dashoffset', initialOffset);
            progressTube.setAttribute('stroke-linecap', 'round');
            progressTube.setAttribute('data-metric-idx', idx);
            progressTube.style.transition = 'stroke-dashoffset 2s cubic-bezier(0.4, 0.0, 0.2, 1)';
            progressTube.style.willChange = 'stroke-dashoffset';
            g.appendChild(progressTube);

            // Top highlight for 3D effect
            const highlight = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            highlight.setAttribute('cx', centerX);
            highlight.setAttribute('cy', centerY);
            highlight.setAttribute('r', radius - tubeWidth/3.5);
            highlight.setAttribute('fill', 'none');
            highlight.setAttribute('stroke', metric.color);
            highlight.setAttribute('stroke-width', '2.5');
            highlight.setAttribute('stroke-dasharray', circumference);
            highlight.setAttribute('stroke-dashoffset', initialOffset);
            highlight.setAttribute('stroke-linecap', 'round');
            highlight.setAttribute('opacity', '0.5');
            highlight.setAttribute('data-metric-idx', idx);
            highlight.setAttribute('data-color', metric.color);
            highlight.style.transition = 'stroke-dashoffset 2s cubic-bezier(0.4, 0.0, 0.2, 1)';
            highlight.style.willChange = 'stroke-dashoffset';
            g.appendChild(highlight);

            // Outer glow effect
            const glow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            glow.setAttribute('cx', centerX);
            glow.setAttribute('cy', centerY);
            glow.setAttribute('r', radius);
            glow.setAttribute('fill', 'none');
            glow.setAttribute('stroke', metric.color);
            glow.setAttribute('stroke-width', tubeWidth + 2);
            glow.setAttribute('stroke-dasharray', circumference);
            glow.setAttribute('stroke-dashoffset', initialOffset);
            glow.setAttribute('stroke-linecap', 'round');
            glow.setAttribute('opacity', '0.15');
            glow.setAttribute('filter', `url(#glow-${idx})`);
            glow.setAttribute('data-metric-idx', idx);
            glow.style.transition = 'stroke-dashoffset 2s cubic-bezier(0.4, 0.0, 0.2, 1)';
            glow.style.willChange = 'stroke-dashoffset';
            g.appendChild(glow);

            svg.appendChild(g);
        });

        svgContainer.appendChild(svg);
        container.appendChild(svgContainer);

        // Legend Grid - horizontal layout at bottom
        const legendGrid = document.createElement('div');
        legendGrid.style.cssText = `
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        `;

        metrics.forEach((metric, idx) => {
            const legendCard = document.createElement('div');
            legendCard.style.cssText = `
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(30, 41, 59, 0.4);
                border-radius: 0.375rem;
                padding: 0.5rem 0.75rem;
                border: 1px solid rgba(71, 85, 105, 0.3);
                backdrop-filter: blur(4px);
            `;

            const dot = document.createElement('div');
            dot.style.cssText = `
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: ${metric.color};
                box-shadow: 0 0 10px ${metric.color}80;
                flex-shrink: 0;
            `;

            const textContainer = document.createElement('div');
            textContainer.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 0.125rem;
            `;

            const label = document.createElement('div');
            label.style.cssText = `
                font-size: 0.625rem;
                color: #94a3b8;
                line-height: 1.2;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            `;
            label.textContent = metric.label || '';

            const value = document.createElement('div');
            value.style.cssText = `
                font-size: 1rem;
                font-weight: 700;
                color: #ffffff;
                font-family: Inter, sans-serif;
                line-height: 1.2;
            `;
            value.textContent = metric.rawValue !== null && metric.rawValue !== undefined
                ? (idx === 2 ? `${metric.rawValue.toFixed(1)} GB` : `${metric.rawValue.toFixed(1)}%`)
                : 'N/A';

            textContainer.appendChild(label);
            textContainer.appendChild(value);
            legendCard.appendChild(dot);
            legendCard.appendChild(textContainer);
            legendGrid.appendChild(legendCard);
        });

        container.appendChild(legendGrid);

        // Animate the circles with staggered delay
        metrics.forEach((metric, idx) => {
            setTimeout(() => {
                const radius = baseRadius + (idx * (tubeWidth + gap));
                const circumference = 2 * Math.PI * radius;
                const finalOffset = circumference - (metric.value / 100) * circumference;

                // Update progress tube
                const progressTube = svg.querySelector(`circle[data-metric-idx="${idx}"][stroke*="gradient"]`);
                if (progressTube) {
                    progressTube.setAttribute('stroke-dashoffset', finalOffset);
                }

                // Update highlight (first circle with matching color and data attribute)
                const highlight = svg.querySelector(`circle[data-metric-idx="${idx}"][data-color="${metric.color}"]`);
                if (highlight) {
                    highlight.setAttribute('stroke-dashoffset', finalOffset);
                }

                // Update glow
                const glow = svg.querySelector(`circle[data-metric-idx="${idx}"][filter*="glow"]`);
                if (glow) {
                    glow.setAttribute('stroke-dashoffset', finalOffset);
                }
            }, idx * 150);
        });

        // Store references for updates
        charts.resourceUtilization = {
            svg: svg,
            metrics: metrics,
            baseRadius: baseRadius,
            tubeWidth: tubeWidth,
            gap: gap,
            centerX: centerX,
            centerY: centerY,
            legendGrid: legendGrid,
            update: function(newMetrics) {
                newMetrics.forEach((metric, idx) => {
                    const radius = this.baseRadius + (idx * (this.tubeWidth + this.gap));
                    const circumference = 2 * Math.PI * radius;
                    const finalOffset = circumference - (metric.value / 100) * circumference;

                    // Update progress tube
                    const progressTube = this.svg.querySelector(`circle[data-metric-idx="${idx}"][stroke*="gradient"]`);
                    if (progressTube) {
                        progressTube.setAttribute('stroke-dashoffset', finalOffset);
                    }

                    // Update highlight
                    const highlight = this.svg.querySelector(`circle[data-metric-idx="${idx}"][data-color="${metric.color}"]`);
                    if (highlight) {
                        highlight.setAttribute('stroke-dashoffset', finalOffset);
                    }

                    // Update glow
                    const glow = this.svg.querySelector(`circle[data-metric-idx="${idx}"][filter*="glow"]`);
                    if (glow) {
                        glow.setAttribute('stroke-dashoffset', finalOffset);
                    }

                    // Update legend label and value
                    if (this.legendGrid && this.legendGrid.children[idx]) {
                        const textContainer = this.legendGrid.children[idx].querySelector('div:last-child');
                        if (textContainer) {
                            const legendLabel = textContainer.querySelector('div:first-child');
                            const legendValue = textContainer.querySelector('div:last-child');

                            if (legendLabel && metric.label) {
                                legendLabel.textContent = metric.label;
                            }

                            if (legendValue) {
                                legendValue.textContent = metric.rawValue !== null && metric.rawValue !== undefined
                                    ? (idx === 2 ? `${metric.rawValue.toFixed(1)} GB` : `${metric.rawValue.toFixed(1)}%`)
                                    : 'N/A';
                            }
                        }
                    }
                });
                this.metrics = newMetrics;
            }
        };

    } catch (error) {
        console.error('Failed to render resource utilization metrics:', error);
    }
}


function updateReliabilityMetrics() {
    updateElement('relDeterminism', metricsData.reliability.determinismScore, 'percentage');
    updateElement('relStability', metricsData.reliability.outputStability, 'percentage');
    updateElement('relValidity', metricsData.reliability.outputValidity, 'percentage');
    updateElement('relCrash', metricsData.reliability.crashRate, 'percentage');
    updateElement('relRetry', metricsData.reliability.retryRate, 'percentage');
    updateElement('relCompletion', metricsData.reliability.completionSuccess, 'percentage');
    updateElement('relSchema', metricsData.reliability.schemaCompliance, 'percentage');
    updateElement('relJSON', metricsData.reliability.jsonValidity, 'percentage');
    updateElement('relType', metricsData.reliability.typeCorrectness, 'percentage');

    // Calculate overall reliability score if we have data
    const reliabilityMetrics = [
        metricsData.reliability.completionSuccess,
        metricsData.reliability.outputStability,
        metricsData.reliability.outputValidity,
        metricsData.reliability.schemaCompliance
    ].filter(v => v !== null);
    const reliabilityScore = reliabilityMetrics.length > 0
        ? (reliabilityMetrics.reduce((a, b) => a + b, 0) / reliabilityMetrics.length).toFixed(1)
        : null;
    updateElement('overallReliability', reliabilityScore, 'percentage');
}

function updateAgentMetrics() {
    updateElement('agentTaskCompletion', metricsData.agent.taskCompletion, 'percentage');
    updateElement('agentStepEfficiency', metricsData.agent.stepEfficiency, 'percentage');
    updateElement('agentErrorRecovery', metricsData.agent.errorRecovery, 'percentage');
    updateElement('agentToolUsage', metricsData.agent.toolUsageAccuracy, 'percentage');
    updateElement('agentPlanning', metricsData.agent.planningCoherence, 'percentage');
    updateElement('agentActionHall', metricsData.agent.actionHallucination, 'percentage');
    updateElement('agentGoalDrift', metricsData.agent.goalDrift, 'percentage');

    // Calculate overall agent performance if we have data
    const agentMetrics = [
        metricsData.agent.taskCompletion,
        metricsData.agent.stepEfficiency,
        metricsData.agent.errorRecovery,
        metricsData.agent.toolUsageAccuracy,
        metricsData.agent.planningCoherence
    ].filter(v => v !== null);
    const agentScore = agentMetrics.length > 0
        ? (agentMetrics.reduce((a, b) => a + b, 0) / agentMetrics.length).toFixed(1)
        : null;
    updateElement('overallAgent', agentScore, 'percentage');

    // Update progress bars - reset to 0% if no data, otherwise update with actual value
    if (metricsData.agent.taskCompletion !== null && metricsData.agent.taskCompletion !== undefined) {
        updateProgressBarWidth('agentTaskCompletion', metricsData.agent.taskCompletion);
    } else {
        updateProgressBarWidth('agentTaskCompletion', 0);
    }
    if (metricsData.agent.stepEfficiency !== null && metricsData.agent.stepEfficiency !== undefined) {
        updateProgressBarWidth('agentStepEfficiency', metricsData.agent.stepEfficiency);
    } else {
        updateProgressBarWidth('agentStepEfficiency', 0);
    }
    if (metricsData.agent.errorRecovery !== null && metricsData.agent.errorRecovery !== undefined) {
        updateProgressBarWidth('agentErrorRecovery', metricsData.agent.errorRecovery);
    } else {
        updateProgressBarWidth('agentErrorRecovery', 0);
    }

    // Update Planning & Execution progress bars
    if (metricsData.agent.toolUsageAccuracy !== null && metricsData.agent.toolUsageAccuracy !== undefined) {
        updateProgressBarWidth('agentToolUsageProgress', metricsData.agent.toolUsageAccuracy);
    } else {
        updateProgressBarWidth('agentToolUsageProgress', 0);
    }
    if (metricsData.agent.planningCoherence !== null && metricsData.agent.planningCoherence !== undefined) {
        updateProgressBarWidth('agentPlanningProgress', metricsData.agent.planningCoherence);
    } else {
        updateProgressBarWidth('agentPlanningProgress', 0);
    }
    if (metricsData.agent.actionHallucination !== null && metricsData.agent.actionHallucination !== undefined) {
        updateProgressBarWidth('agentActionHallProgress', metricsData.agent.actionHallucination);
    } else {
        updateProgressBarWidth('agentActionHallProgress', 0);
    }
    if (metricsData.agent.goalDrift !== null && metricsData.agent.goalDrift !== undefined) {
        updateProgressBarWidth('agentGoalDriftProgress', metricsData.agent.goalDrift);
    } else {
        updateProgressBarWidth('agentGoalDriftProgress', 0);
    }
}

function updateSecurityMetrics(stats) {
    updateElement('secInjection', metricsData.security.injectionAttackSuccess, 'percentage');
    updateElement('secAdversarial', metricsData.security.adversarialVulnerability, 'percentage');
    updateElement('secExfiltration', metricsData.security.dataExfiltration, 'percentage');
    updateElement('secEvasion', metricsData.security.modelEvasion, 'percentage');
    updateElement('secExtraction', metricsData.security.extractionRisk, 'percentage');

    // Attack counts - get from statistics
    const securityTestCounts = stats?.security_test_counts || {};
    updateElement('attackInjection', securityTestCounts.injection || 0);
    updateElement('attackAdversarial', securityTestCounts.adversarial || 0);
    updateElement('attackExfiltration', securityTestCounts.exfiltration || 0);
    updateElement('attackExtraction', securityTestCounts.extraction || 0);

    // Calculate overall security score if we have data
    // Note: 0.0 is a valid value (no vulnerability), so check for null/undefined explicitly
    const securityMetrics = [
        metricsData.security.injectionAttackSuccess !== null && metricsData.security.injectionAttackSuccess !== undefined
            ? 100 - metricsData.security.injectionAttackSuccess : null,
        metricsData.security.adversarialVulnerability !== null && metricsData.security.adversarialVulnerability !== undefined
            ? 100 - metricsData.security.adversarialVulnerability : null,
        metricsData.security.dataExfiltration !== null && metricsData.security.dataExfiltration !== undefined
            ? 100 - metricsData.security.dataExfiltration : null,
        metricsData.security.modelEvasion !== null && metricsData.security.modelEvasion !== undefined
            ? 100 - metricsData.security.modelEvasion : null,
        metricsData.security.extractionRisk !== null && metricsData.security.extractionRisk !== undefined
            ? 100 - metricsData.security.extractionRisk : null
    ].filter(v => v !== null && v !== undefined);
    const securityScore = securityMetrics.length > 0
        ? (securityMetrics.reduce((a, b) => a + b, 0) / securityMetrics.length).toFixed(1)
        : null;
    updateElement('overallSecurity', securityScore, 'percentage');
}

function updateFooterStats(stats) {
    updateElement('footerTotalTests', (stats.total_tests || 0).toLocaleString());

    // Metrics tracked - count unique metric names from database
    const metricsTracked = stats.metrics_tracked || 0;
    updateElement('footerMetrics', metricsTracked > 0 ? metricsTracked.toString() : null);

    // Data points - total records across all metric tables
    const dataPoints = stats.data_points || 0;
    let dataPointsDisplay = null;
    if (dataPoints > 0) {
        if (dataPoints >= 1000000) {
            dataPointsDisplay = `${(dataPoints / 1000000).toFixed(1)}M`;
        } else if (dataPoints >= 1000) {
            dataPointsDisplay = `${(dataPoints / 1000).toFixed(1)}K`;
        } else {
            dataPointsDisplay = dataPoints.toLocaleString();
        }
    }
    updateElement('footerDataPoints', dataPointsDisplay);
}

function updateElement(id, value, format = null) {
    const element = document.getElementById(id);
    if (!element) return;

    if (value === null || value === undefined) {
        element.textContent = 'N/A';
        element.style.opacity = '0.5';
        element.style.fontStyle = 'italic';
    } else {
        element.style.opacity = '1';
        element.style.fontStyle = 'normal';
        if (format === 'percentage') {
            // Round to 1 decimal place for percentages
            const roundedValue = typeof value === 'number' ? parseFloat(value.toFixed(1)) : value;
            element.textContent = `${roundedValue}%`;
        } else if (format === 'decimal') {
            element.textContent = typeof value === 'number' ? value.toFixed(2) : value;
        } else {
            element.textContent = value;
        }
    }
}

function updateProgressBarWidth(elementId, percentage) {
    const element = document.getElementById(elementId);
    if (element) {
        // Direct update if element is the progress bar itself
        if (element.classList && element.classList.contains('progress-fill')) {
            element.style.width = `${percentage}%`;
            return;
        }
        // If element is the progress bar container, find the fill
        const progressBar = element.querySelector ? element.querySelector('.progress-fill') : null;
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            return;
        }
        // Find the closest parent with a progress bar
        const container = element.closest ? element.closest('.rag-metric-item, .agent-metric-item, .utilization-item, .progress-bar-container') : null;
        if (container) {
            const fill = container.querySelector('.progress-fill');
            if (fill) {
                fill.style.width = `${percentage}%`;
            }
        }
    }
}

// ===== CHARTS =====
async function renderOverviewCharts() {
    await renderPerformanceTrendChart();
    renderHealthRadarChart();
    renderCategoryBreakdown();
}

async function renderPerformanceTrendChart() {
    const container = document.getElementById('performanceTrendChart');
    const footerContainer = document.getElementById('performanceTrendsFooter');
    if (!container || typeof ApexCharts === 'undefined') return;

    try {
        // Fetch real trend data
        const response = await fetch('/api/trends?hours=24');
        const trends = await response.json();

        let data = [];
        let peakAccuracy = 0;
        let avgAccuracy = 0;
        let minLatency = Infinity;
        let maxLatency = 0;
        let avgLatency = 0;
        let trendValue = 0;
        let latencyTrend = 0;
        let dataPoints = 0;

        if (trends.test_results && trends.test_results.length > 0) {
            // Use more data points for better visualization (last 12 points or all if less)
            const results = trends.test_results.slice(-12);
            const now = new Date();
            const isToday = (date) => {
                const d = new Date(date);
                return d.toDateString() === now.toDateString();
            };

            data = results.map(result => {
                const date = new Date(result.timestamp);
                // Format time: show date if not today, otherwise just time
                let timeLabel;
                if (isToday(result.timestamp)) {
                    timeLabel = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
                } else {
                    timeLabel = `${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
                }

                const accuracy = result.pass_rate !== null && result.pass_rate !== undefined ? result.pass_rate : null;
                // Calculate latency from duration (convert to ms)
                const latency = result.avg_duration !== null && result.avg_duration !== undefined
                    ? Math.round(result.avg_duration * 1000)
                    : null;

                if (accuracy !== null) {
                    peakAccuracy = Math.max(peakAccuracy, accuracy);
                    avgAccuracy += accuracy;
                    dataPoints++;
                }

                if (latency !== null) {
                    minLatency = Math.min(minLatency, latency);
                    maxLatency = Math.max(maxLatency, latency);
                    avgLatency += latency;
                }

                return {
                    time: timeLabel,
                    timestamp: date.getTime(),
                    accuracy: accuracy,
                    latency: latency
                };
            }).filter(d => d.accuracy !== null || d.latency !== null);

            // Calculate averages
            if (dataPoints > 0) {
                avgAccuracy = avgAccuracy / dataPoints;
            }
            const latencyPoints = data.filter(d => d.latency !== null).length;
            if (latencyPoints > 0) {
                avgLatency = avgLatency / latencyPoints;
            }

            // Calculate trends (compare first and last)
            if (data.length >= 2) {
                const firstAccuracy = data.find(d => d.accuracy !== null)?.accuracy;
                const lastAccuracy = data.slice().reverse().find(d => d.accuracy !== null)?.accuracy;
                if (firstAccuracy !== null && firstAccuracy !== undefined && firstAccuracy > 0 &&
                    lastAccuracy !== null && lastAccuracy !== undefined) {
                    trendValue = parseFloat(((lastAccuracy - firstAccuracy) / firstAccuracy * 100).toFixed(1));
                }

                const firstLatency = data.find(d => d.latency !== null)?.latency;
                const lastLatency = data.slice().reverse().find(d => d.latency !== null)?.latency;
                if (firstLatency !== null && firstLatency !== undefined && firstLatency > 0 &&
                    lastLatency !== null && lastLatency !== undefined) {
                    latencyTrend = parseFloat(((firstLatency - lastLatency) / firstLatency * 100).toFixed(1)); // Inverted: lower is better
                }
            }
        }

        // If no data, show empty chart with message
        if (data.length === 0) {
            container.innerHTML = '<div style="text-align: center; padding: 3rem; color: var(--text-secondary);">No trend data available yet. Run tests to see performance trends.</div>';
            if (footerContainer) footerContainer.innerHTML = '';
            return;
        }

        if (charts.performanceTrend) {
            charts.performanceTrend.destroy();
        }

        // Prepare data for dual y-axis chart
        // Ensure data arrays match categories exactly - use null for missing values to maintain alignment
        const accuracyData = data.map(d => d.accuracy !== null && d.accuracy !== undefined ? d.accuracy : null);
        const latencyData = data.map(d => d.latency !== null && d.latency !== undefined ? d.latency : null);
        const categories = data.map(d => d.time);

        // Only animate on first render
        const shouldAnimate = !chartAnimationsPlayed.performanceTrend;
        if (shouldAnimate) {
            chartAnimationsPlayed.performanceTrend = true;
        }

        // Use a fixed, balanced height that matches the health radar chart
        // This ensures both charts look proportional and balanced
        const calculatedHeight = 450;

        charts.performanceTrend = new ApexCharts(container, {
            series: [
                {
                    name: 'Accuracy',
                    type: 'area',
                    data: accuracyData
                },
                {
                    name: 'Latency',
                    type: 'area',
                    data: latencyData
                }
            ],
            chart: {
                type: 'area',
                height: calculatedHeight,
                background: 'transparent',
                toolbar: { show: false },
                zoom: { enabled: false },
                fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                animations: {
                    enabled: shouldAnimate,
                    easing: 'easeinout',
                    speed: shouldAnimate ? 1800 : 0,
                    animateGradually: {
                        enabled: shouldAnimate,
                        delay: shouldAnimate ? 250 : 0
                    }
                },
                dropShadow: {
                    enabled: true,
                    color: ['rgba(96, 165, 250, 0.4)', 'rgba(245, 158, 11, 0.4)'],
                    blur: 12,
                    opacity: 0.5,
                    top: 8,
                    left: 0
                }
            },
            xaxis: {
                categories: categories,
                type: 'category',
                labels: {
                    style: {
                        colors: '#94a3b8',
                        fontSize: '11px',
                        fontWeight: 500,
                        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                    },
                    rotate: 0,
                    rotateAlways: false,
                    hideOverlappingLabels: true,
                    trim: true,
                    maxHeight: 40,
                    offsetY: 8
                },
                axisBorder: {
                    show: false
                },
                axisTicks: {
                    show: false
                },
                tooltip: {
                    enabled: false
                }
            },
            yaxis: [
                {
                    title: {
                        text: 'Accuracy (%)',
                        style: {
                            color: '#60a5fa',
                            fontSize: '12px',
                            fontWeight: 700,
                            fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                        },
                        offsetX: 0,
                        offsetY: 0
                    },
                    min: 0,
                    max: 100,
                    labels: {
                        style: {
                            colors: '#60a5fa',
                            fontSize: '11px',
                            fontWeight: 600,
                            fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                        },
                        formatter: function(val) {
                            return Math.round(val) + '%';
                        },
                        offsetX: -5
                    },
                    axisBorder: {
                        show: false
                    }
                },
                {
                    opposite: true,
                    title: {
                        text: 'Latency (ms)',
                        style: {
                            color: '#f59e0b',
                            fontSize: '12px',
                            fontWeight: 700,
                            fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                        },
                        offsetX: 0,
                        offsetY: 0
                    },
                    min: 0,
                    labels: {
                        style: {
                            colors: '#f59e0b',
                            fontSize: '11px',
                            fontWeight: 600,
                            fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                        },
                        formatter: function(val) {
                            if (val >= 1000) {
                                return (val / 1000).toFixed(1) + 'k';
                            }
                            return Math.round(val);
                        },
                        offsetX: 5
                    },
                    axisBorder: {
                        show: false
                    }
                }
            ],
            colors: ['#60a5fa', '#f59e0b'],
            stroke: {
                width: [4, 4],
                curve: 'smooth',
                lineCap: 'round',
                colors: ['#3b82f6', '#f59e0b']
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'dark',
                    type: 'vertical',
                    shadeIntensity: 1,
                    gradientToColors: ['#3b82f6', '#d97706'],
                    inverseColors: false,
                    opacityFrom: [0.8, 0.7],
                    opacityTo: [0.2, 0.15],
                    stops: [0, 50, 100]
                }
            },
            markers: {
                size: [0, 0],
                strokeWidth: 0,
                hover: {
                    size: 0
                },
                showNullDataPoints: false
            },
            dataLabels: {
                enabled: false
            },
            legend: {
                show: true,
                position: 'top',
                offsetY: 0,
                horizontalAlign: 'right',
                floating: false,
                fontSize: '12px',
                fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                fontWeight: 600,
                labels: {
                    colors: '#e2e8f0',
                    useSeriesColors: true
                },
                markers: {
                    width: 14,
                    height: 14,
                    radius: 7,
                    offsetX: -7,
                    offsetY: 0
                },
                itemMargin: {
                    horizontal: 28,
                    vertical: 8
                }
            },
            grid: {
                borderColor: 'rgba(71, 85, 105, 0.15)',
                strokeDashArray: 4,
                xaxis: {
                    lines: {
                        show: false
                    }
                },
                yaxis: {
                    lines: {
                        show: true,
                        strokeDashArray: 4
                    }
                },
                padding: {
                    top: 30,
                    right: 15,
                    left: 10,
                    bottom: 15
                }
            },
            tooltip: {
                theme: 'dark',
                shared: true,
                intersect: false,
                followCursor: true,
                x: {
                    show: true,
                    format: 'dd MMM HH:mm'
                },
                y: {
                    formatter: function(val, { seriesIndex }) {
                        if (val === null || val === undefined) return 'N/A';
                        if (seriesIndex === 0) {
                            return val.toFixed(1) + '%';
                        } else {
                            if (val >= 1000) {
                                return (val / 1000).toFixed(1) + 'k ms';
                            }
                            return Math.round(val) + ' ms';
                        }
                    }
                },
                marker: {
                    show: true
                },
                style: {
                    fontSize: '12px',
                    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                }
            },
            theme: {
                mode: 'dark',
                palette: 'palette1'
            }
        });

        charts.performanceTrend.render();

        // Chart height is now fixed, so no need for dynamic resize handler
        // The CSS will handle responsive sizing with max-height

        // Render footer with enhanced stats
        if (footerContainer) {
            const footerHtml = `
                <div class="performance-trends-footer-content">
                    <div class="performance-trends-stats">
                        <div class="performance-stat-item">
                            <div class="performance-stat-indicator performance-stat-indicator-blue"></div>
                            <div class="performance-stat-group">
                                <span class="performance-stat-label">Peak Accuracy</span>
                                <span class="performance-stat-value-blue">${peakAccuracy.toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="performance-stat-item">
                            <div class="performance-stat-indicator performance-stat-indicator-amber"></div>
                            <div class="performance-stat-group">
                                <span class="performance-stat-label">Avg Latency</span>
                                <span class="performance-stat-value-amber">${avgLatency > 0 ? Math.round(avgLatency) + 'ms' : 'N/A'}</span>
                            </div>
                        </div>
                        <div class="performance-stat-item">
                            <div class="performance-stat-indicator performance-stat-indicator-blue"></div>
                            <div class="performance-stat-group">
                                <span class="performance-stat-label">Avg Accuracy</span>
                                <span class="performance-stat-value-blue">${avgAccuracy > 0 ? avgAccuracy.toFixed(1) + '%' : 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    <div class="performance-trend-indicator">
                        <div class="performance-trend-dot"></div>
                        <div class="performance-trend-group">
                            <span class="performance-trend-label">Accuracy Trend</span>
                            <span class="performance-trend-text ${trendValue >= 0 ? 'trend-positive' : 'trend-negative'}">${trendValue >= 0 ? '+' : ''}${trendValue}%</span>
                        </div>
                        ${latencyTrend !== 0 ? `
                        <div class="performance-trend-group" style="margin-top: 4px;">
                            <span class="performance-trend-label">Latency Trend</span>
                            <span class="performance-trend-text ${latencyTrend >= 0 ? 'trend-positive' : 'trend-negative'}">${latencyTrend >= 0 ? '+' : ''}${latencyTrend}%</span>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
            footerContainer.innerHTML = footerHtml;
        }
    } catch (error) {
        console.error('Failed to load trend data:', error);
        container.innerHTML = '<div style="text-align: center; padding: 3rem; color: var(--text-secondary);">Failed to load trend data.</div>';
        if (footerContainer) footerContainer.innerHTML = '';
    }
}

function renderHealthRadarChart() {
    const container = document.getElementById('healthRadarChart');
    const footerContainer = document.getElementById('healthRadarFooter');
    const scoreBadge = document.getElementById('healthScoreBadge');
    if (!container || typeof ApexCharts === 'undefined') return;

    // Calculate values from real metrics (convert to 0-100 scale)
    // Use explicit null checks to avoid falsy 0 values
    let accuracy = metricsData.baseModel.accuracy;
    if (accuracy === null || accuracy === undefined) {
        if (metricsData.baseModel.bertScore !== null && metricsData.baseModel.bertScore !== undefined) {
            accuracy = metricsData.baseModel.bertScore * 100;
        } else if (metricsData.baseModel.rougeL !== null && metricsData.baseModel.rougeL !== undefined) {
            accuracy = metricsData.baseModel.rougeL * 100;
        } else {
            accuracy = null;
        }
    }

    // IMPORTANT: Include injection attack success (inverted) as it's a critical safety metric
    const safetyComponents = [];
    if (metricsData.safety.complianceScore !== null) {
        safetyComponents.push(metricsData.safety.complianceScore);
    }
    if (metricsData.safety.toxicityScore !== null) {
        safetyComponents.push(100 - metricsData.safety.toxicityScore);
    }
    if (metricsData.safety.harmfulnessScore !== null) {
        safetyComponents.push(100 - metricsData.safety.harmfulnessScore);
    }
    // Include injection attack success (inverted - lower attack success = higher safety)
    if (metricsData.security && metricsData.security.injectionAttackSuccess !== null) {
        safetyComponents.push(100 - metricsData.security.injectionAttackSuccess);
    }
    // Include prompt injection resistance (inverted - lower injection = higher safety)
    if (metricsData.safety.promptInjection !== null) {
        safetyComponents.push(100 - metricsData.safety.promptInjection);
    }
    const safety = safetyComponents.length > 0
        ? (safetyComponents.reduce((a, b) => a + b, 0) / safetyComponents.length)
        : null;

    // Gold Context Match (RAG metric - NOT displayed elsewhere in overview)
    const goldContextMatch = metricsData.rag.goldContextMatch !== null && metricsData.rag.goldContextMatch !== undefined
        ? metricsData.rag.goldContextMatch
        : null;

    // Reranker Score (RAG metric - NOT displayed elsewhere in overview)
    // Note: rerankerScore comes as 0-1 decimal from backend, convert to 0-100 percentage
    let rerankerScore = null;
    if (metricsData.rag.rerankerScore !== null && metricsData.rag.rerankerScore !== undefined) {
        // If value is <= 1, assume it's a decimal (0-1) and convert to percentage (0-100)
        // If value is > 1, assume it's already a percentage
        rerankerScore = metricsData.rag.rerankerScore <= 1
            ? metricsData.rag.rerankerScore * 100
            : metricsData.rag.rerankerScore;
    }

    // Refusal Rate (Safety metric - NOT displayed elsewhere in overview, inverted for health)
    const refusalRate = metricsData.safety.refusalRate !== null && metricsData.safety.refusalRate !== undefined
        ? Math.max(0, 100 - metricsData.safety.refusalRate) // Invert: lower refusal = higher health
        : null;

    // Bias Score (Safety metric - NOT displayed elsewhere in overview, inverted for health)
    const biasScore = metricsData.safety.biasScore !== null && metricsData.safety.biasScore !== undefined
        ? Math.max(0, 100 - metricsData.safety.biasScore) // Invert: lower bias = higher health
        : null;

    // Determinism Score (Reliability metric - NOT displayed elsewhere in overview)
    const determinismScore = metricsData.reliability.determinismScore !== null && metricsData.reliability.determinismScore !== undefined
        ? metricsData.reliability.determinismScore
        : null;

    // Task Completion (Agent metric - NOT displayed elsewhere in overview)
    const taskCompletion = metricsData.agent.taskCompletion !== null && metricsData.agent.taskCompletion !== undefined
        ? metricsData.agent.taskCompletion
        : null;

    // Step Efficiency (Agent metric - NOT displayed elsewhere in overview)
    const stepEfficiency = metricsData.agent.stepEfficiency !== null && metricsData.agent.stepEfficiency !== undefined
        ? metricsData.agent.stepEfficiency
        : null;

    const data = [
        { metric: 'Gold Match', value: goldContextMatch },
        { metric: 'Reranker', value: rerankerScore },
        { metric: 'Refusal', value: refusalRate },
        { metric: 'Bias', value: biasScore },
        { metric: 'Determinism', value: determinismScore },
        { metric: 'Efficiency', value: stepEfficiency }
    ].filter(d => {
        // More robust filtering - ensure value is a valid number
        const isValid = d.value !== null &&
                       d.value !== undefined &&
                       !isNaN(d.value) &&
                       isFinite(d.value);
        return isValid;
    });

    // Calculate overall health score
    const validValues = data.map(d => d.value).filter(v => v !== null && v !== undefined);
    const overallScore = validValues.length > 0
        ? (validValues.reduce((a, b) => a + b, 0) / validValues.length).toFixed(1)
        : null;

    // Update score badge
    if (scoreBadge && overallScore) {
        scoreBadge.innerHTML = `<p class="health-score-text">${overallScore}/100</p>`;
    }

    // If no data available, show message
    if (data.length === 0) {
        container.innerHTML = '<div style="text-align: center; padding: 3rem; color: var(--text-secondary);">No health data available yet. Run tests to see multi-dimensional health metrics.</div>';
        if (footerContainer) footerContainer.innerHTML = '';
        return;
    }

    if (charts.healthRadar) {
        charts.healthRadar.destroy();
    }

    // Find highest, average, and lowest
    const values = data.map(d => d.value);
    const highest = Math.max(...values);
    const lowest = Math.min(...values);
    const highestMetric = data.find(d => d.value === highest);
    const lowestMetric = data.find(d => d.value === lowest);

    // Only animate on first render
    const shouldAnimate = !chartAnimationsPlayed.healthRadar;
    if (shouldAnimate) {
        chartAnimationsPlayed.healthRadar = true;
    }

    charts.healthRadar = new ApexCharts(container, {
        series: [{
            name: 'System Health',
            data: data.map(d => d.value)
        }],
        chart: {
            type: 'radar',
            height: 450,
            background: 'transparent',
            toolbar: { show: false },
            animations: {
                enabled: shouldAnimate,
                easing: 'easeinout',
                speed: shouldAnimate ? 1000 : 0
            }
        },
        xaxis: {
            categories: data.map(d => d.metric),
            labels: {
                style: {
                    colors: '#cbd5e1',
                    fontSize: '13px',
                    fontWeight: 600
                }
            }
        },
        yaxis: {
            min: 0,
            max: 100,
            labels: {
                style: {
                    colors: '#94a3b8',
                    fontSize: '11px'
                }
            }
        },
        colors: ['#8b5cf6'],
        fill: {
            opacity: 0.6,
            type: 'gradient',
            gradient: {
                shade: 'dark',
                type: 'diagonal1',
                shadeIntensity: 0.5,
                gradientToColors: ['#a78bfa'],
                inverseColors: false,
                opacityFrom: 0.6,
                opacityTo: 0.2,
                stops: [0, 50, 100]
            }
        },
        stroke: {
            width: 3,
            colors: ['#8b5cf6']
        },
        markers: {
            size: 5,
            strokeWidth: 3,
            strokeColors: ['#5b21b6'],
            fillColors: ['#8b5cf6'],
            hover: {
                size: 7,
                sizeOffset: 2
            }
        },
        plotOptions: {
            radar: {
                polygons: {
                    strokeColors: '#475569',
                    strokeWidth: 1.5,
                    connectorColors: '#475569',
                    fill: {
                        colors: undefined
                    }
                }
            }
        },
        tooltip: {
            theme: 'dark',
            custom: function({ series, seriesIndex, dataPointIndex, w }) {
                const value = series[seriesIndex][dataPointIndex];
                const metric = w.globals.categoryLabels[dataPointIndex] || 'Unknown';

                // Safety check for invalid values
                if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
                    return '';
                }

                const getStatus = (val) => {
                    if (val >= 90) return { text: 'Excellent', color: '#10b981', icon: '🌟' };
                    if (val >= 80) return { text: 'Good', color: '#3b82f6', icon: '✓' };
                    if (val >= 70) return { text: 'Fair', color: '#f59e0b', icon: '⚠' };
                    return { text: 'Needs Attention', color: '#ef4444', icon: '⚠' };
                };

                const status = getStatus(value);

                return `
                    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(139, 92, 246, 0.5); border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 20px rgba(139, 92, 246, 0.3); backdrop-filter: blur(20px); padding: 12px 16px;">
                        <div style="color: #8b5cf6; font-weight: 700; font-size: 16px; margin-bottom: 4px;">
                            ${value.toFixed(1)}/100
                        </div>
                        <div style="color: ${status.color}; font-size: 12px; font-weight: 600;">
                            ${status.icon} ${status.text}
                        </div>
                        <div style="color: #94a3b8; font-size: 11px; margin-top: 4px;">
                            ${metric}
                        </div>
                    </div>
                `;
            },
            style: {
                fontSize: '12px'
            }
        },
        theme: { mode: 'dark' }
    });

    charts.healthRadar.render();

    // Render footer with stats
    if (footerContainer && highestMetric && lowestMetric && overallScore) {
        const footerHtml = `
            <div class="health-radar-footer-content">
                <div class="health-stat-card">
                    <p class="health-stat-label">Highest</p>
                    <p class="health-stat-value health-stat-value-emerald">${highestMetric.metric} ${highest.toFixed(1)}%</p>
                </div>
                <div class="health-stat-card">
                    <p class="health-stat-label">Average</p>
                    <p class="health-stat-value health-stat-value-purple">Overall ${overallScore}%</p>
                </div>
                <div class="health-stat-card">
                    <p class="health-stat-label">Lowest</p>
                    <p class="health-stat-value health-stat-value-amber">${lowestMetric.metric} ${lowest.toFixed(1)}%</p>
                </div>
            </div>
        `;
        footerContainer.innerHTML = footerHtml;
    }
}

function renderCategoryBreakdown() {
    const container = document.getElementById('categoryBreakdown');
    const summaryContainer = document.getElementById('categoryBreakdownSummary');
    const avgScoreElement = document.getElementById('categoryAvgScore');
    if (!container) return;

    // Calculate category scores from real metrics
    // Note: bertScore and rougeL are already converted to percentages (0-100) in calculateMetricsFromStats()
    const baseModelMetrics = [
        metricsData.baseModel.accuracy,
        metricsData.baseModel.bertScore,
        metricsData.baseModel.rougeL
    ].filter(v => v !== null && v !== undefined);
    const baseModelScore = baseModelMetrics.length > 0
        ? Math.round(baseModelMetrics.reduce((a, b) => a + b, 0) / baseModelMetrics.length)
        : null;
    const baseModelMetricsCount = baseModelMetrics.length;

    const ragMetrics = [
        metricsData.rag.retrievalRecall5,
        metricsData.rag.contextRelevance,
        metricsData.rag.contextCoverage
    ].filter(v => v !== null && v !== undefined);
    const ragScore = ragMetrics.length > 0
        ? Math.round(ragMetrics.reduce((a, b) => a + b, 0) / ragMetrics.length)
        : null;
    const ragMetricsCount = ragMetrics.length;

    // Use shared Safety Score calculation function (round to integer for category breakdown)
    const safetyScore = calculateSafetyScore(true); // true = round to integer
    // Count only actual safety metrics (not security metrics)
    const safetyMetricsCount = [
        metricsData.safety.complianceScore,
        metricsData.safety.toxicityScore,
        metricsData.safety.biasScore,
        metricsData.safety.harmfulnessScore,
        metricsData.safety.ethicalViolation,
        metricsData.safety.dataLeakage,
        metricsData.safety.piiLeakage,
        metricsData.safety.promptInjection
    ].filter(v => v !== null && v !== undefined).length;

    // Calculate performance score from multiple metrics
    const performanceMetrics = [];

    // E2E Latency score (inverse: lower is better, target: 2000ms = 100%, 4000ms = 0%)
    if (metricsData.performance.e2eLatency !== null && metricsData.performance.e2eLatency !== undefined) {
        const e2eScore = Math.max(0, Math.min(100, 100 - ((metricsData.performance.e2eLatency - 2000) / 20)));
        performanceMetrics.push(e2eScore);
    }

    // TTFT score (inverse: lower is better, target: 300ms = 100%, 800ms = 0%)
    if (metricsData.performance.ttft !== null && metricsData.performance.ttft !== undefined) {
        const ttftScore = Math.max(0, Math.min(100, 100 - ((metricsData.performance.ttft - 300) / 5)));
        performanceMetrics.push(ttftScore);
    }

    // Throughput score (higher is better, target: 50 tokens/sec = 100%, 10 tokens/sec = 0%)
    if (metricsData.performance.throughput !== null && metricsData.performance.throughput !== undefined) {
        const throughputScore = Math.max(0, Math.min(100, ((metricsData.performance.throughput - 10) / 40) * 100));
        performanceMetrics.push(throughputScore);
    }

    // Token Latency score (inverse: lower is better, target: 50ms = 100%, 200ms = 0%)
    if (metricsData.performance.tokenLatency !== null && metricsData.performance.tokenLatency !== undefined) {
        const tokenLatencyScore = Math.max(0, Math.min(100, 100 - ((metricsData.performance.tokenLatency - 50) / 1.5)));
        performanceMetrics.push(tokenLatencyScore);
    }

    const performanceScore = performanceMetrics.length > 0
        ? Math.round(performanceMetrics.reduce((a, b) => a + b, 0) / performanceMetrics.length)
        : null;
    const performanceMetricsCount = performanceMetrics.length;

    const reliabilityMetrics = [
        metricsData.reliability.completionSuccess,
        metricsData.reliability.outputStability,
        metricsData.reliability.outputValidity
    ].filter(v => v !== null && v !== undefined);
    const reliabilityScore = reliabilityMetrics.length > 0
        ? Math.round(reliabilityMetrics.reduce((a, b) => a + b, 0) / reliabilityMetrics.length)
        : null;
    const reliabilityMetricsCount = reliabilityMetrics.length;

    const agentMetrics = [
        metricsData.agent.taskCompletion,
        metricsData.agent.stepEfficiency,
        metricsData.agent.errorRecovery
    ].filter(v => v !== null && v !== undefined);
    const agentScore = agentMetrics.length > 0
        ? Math.round(agentMetrics.reduce((a, b) => a + b, 0) / agentMetrics.length)
        : null;
    const agentMetricsCount = agentMetrics.length;

    const categories = [
        { name: 'Base Model', value: baseModelScore, color: '#3b82f6', metricsCount: baseModelMetricsCount },
        { name: 'RAG', value: ragScore, color: '#8b5cf6', metricsCount: ragMetricsCount },
        { name: 'Safety', value: safetyScore, color: '#10b981', metricsCount: safetyMetricsCount },
        { name: 'Performance', value: performanceScore, color: '#f59e0b', metricsCount: performanceMetricsCount },
        { name: 'Reliability', value: reliabilityScore, color: '#06b6d4', metricsCount: reliabilityMetricsCount },
        { name: 'Agent', value: agentScore, color: '#ec4899', metricsCount: agentMetricsCount }
    ];

    // Calculate total metrics count (include all categories)
    const totalMetricsCount = categories.reduce((sum, cat) => sum + cat.metricsCount, 0);

    // Calculate average score (only from categories with actual values)
    const categoriesWithValues = categories.filter(cat => cat.value !== null && cat.value !== undefined);
    const avgScore = categoriesWithValues.length > 0
        ? Math.round(categoriesWithValues.reduce((sum, cat) => sum + cat.value, 0) / categoriesWithValues.length)
        : null;

    if (avgScoreElement) {
        avgScoreElement.textContent = avgScore !== null ? `${avgScore}%` : 'N/A';
    }

    // Render categories (show all, even with 0 or N/A)
    container.innerHTML = categories.map((cat, idx) => {
        const circumference = 2 * Math.PI * 50;
        const value = cat.value !== null && cat.value !== undefined ? cat.value : 0;
        const hasValue = cat.value !== null && cat.value !== undefined;
        const displayValue = hasValue ? value : 'N/A';
        const dashArray = hasValue ? `${2 * Math.PI * 50 * value / 100} ${2 * Math.PI * 50}` : `0 ${2 * Math.PI * 50}`;
        const status = !hasValue ? 'N/A' : value >= 90 ? 'Excellent' : value >= 80 ? 'Good' : 'Fair';
        const statusColor = !hasValue ? 'amber' : value >= 90 ? 'emerald' : value >= 80 ? 'blue' : 'amber';
        const statusIcon = !hasValue ? '!' : value >= 90 ? '★' : value >= 80 ? '✓' : '!';

        // Convert hex to RGB for gradient
        const rgb = hexToRgb(cat.color);
        const darkerRgb = {
            r: Math.max(0, Math.floor(rgb.r * 0.6)),
            g: Math.max(0, Math.floor(rgb.g * 0.6)),
            b: Math.max(0, Math.floor(rgb.b * 0.6))
        };

        return `
            <div class="category-breakdown-item">
                <div class="category-breakdown-item-overlay"></div>

                <!-- Circular Progress -->
                <div class="category-breakdown-chart">
                    <svg class="category-breakdown-svg" viewBox="0 0 120 120">
                        <defs>
                            <linearGradient id="category-gradient-${idx}" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="${cat.color}" stop-opacity="1"/>
                                <stop offset="100%" stop-color="${cat.color}" stop-opacity="0.6"/>
                            </linearGradient>
                            <filter id="category-glow-${idx}">
                                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                                <feMerge>
                                    <feMergeNode in="coloredBlur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                            </filter>
                            <filter id="category-shadow-${idx}">
                                <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="${cat.color}" flood-opacity="0.4"/>
                            </filter>
                        </defs>

                        <!-- Background Circle -->
                        <circle
                            cx="60"
                            cy="60"
                            r="50"
                            stroke="#334155"
                            stroke-width="10"
                            fill="none"
                            opacity="0.3"
                        />

                        <!-- Progress Circle -->
                        <circle
                            cx="60"
                            cy="60"
                            r="50"
                            stroke="url(#category-gradient-${idx})"
                            stroke-width="10"
                            fill="none"
                            stroke-dasharray="${dashArray}"
                            stroke-linecap="round"
                            filter="url(#category-glow-${idx})"
                            class="category-progress-circle"
                            transform="rotate(-90 60 60)"
                        />

                        <!-- Animated Shimmer -->
                        <circle
                            cx="60"
                            cy="60"
                            r="50"
                            stroke="${cat.color}"
                            stroke-width="2"
                            fill="none"
                            stroke-dasharray="${dashArray}"
                            stroke-linecap="round"
                            opacity="0.6"
                            filter="url(#category-shadow-${idx})"
                            class="category-shimmer-circle"
                            transform="rotate(-90 60 60)"
                        />
                    </svg>

                    <!-- Center Content -->
                    <div class="category-breakdown-center">
                        <span class="category-breakdown-value" style="color: ${hasValue ? cat.color : '#94a3b8'}">${displayValue}</span>
                        ${hasValue ? '<span class="category-breakdown-max">/ 100</span>' : ''}
                    </div>

                    <!-- Status Indicator -->
                    <div class="category-breakdown-status-badge category-status-${statusColor}">
                        <span class="category-status-icon">${statusIcon}</span>
                    </div>
                </div>

                <!-- Category Info -->
                <div class="category-breakdown-info">
                    <p class="category-breakdown-name">${cat.name}</p>

                    <!-- Performance Bar -->
                    <div class="category-breakdown-bar-bg">
                        ${hasValue ? `<div class="category-breakdown-bar-fill" style="width: ${value}%; background: linear-gradient(90deg, ${cat.color}, ${cat.color}dd); box-shadow: 0 0 10px ${cat.color}40;"></div>` : '<div class="category-breakdown-bar-fill" style="width: 0%; background: linear-gradient(90deg, #64748b, #64748bdd);"></div>'}
                    </div>

                    <!-- Status Label -->
                    <div class="category-breakdown-status-label category-status-label-${statusColor}">
                        <div class="category-status-dot category-status-dot-${statusColor}"></div>
                        ${status}
                    </div>

                    <!-- Metrics Count -->
                    <p class="category-breakdown-metrics-count">${cat.metricsCount} metrics</p>
                </div>
            </div>
        `;
    }).join('');

    // Render summary cards
    if (summaryContainer && categories.length > 0) {
        // Filter categories with actual values for summary calculations
        const categoriesWithValues = categories.filter(cat => cat.value !== null && cat.value !== undefined);

        // Only calculate summary cards if we have actual data
        if (categoriesWithValues.length === 0) {
            // No data available - show N/A for all summary cards
            summaryContainer.innerHTML = `
                <div class="category-summary-content">
                    <div class="category-summary-card category-summary-emerald">
                        <div class="category-summary-header">
                            <p class="category-summary-label">Top Performer</p>
                            <div class="category-summary-icon category-summary-icon-emerald">
                                <i data-lucide="award"></i>
                            </div>
                        </div>
                        <p class="category-summary-title">N/A</p>
                        <p class="category-summary-subtitle">No data available</p>
                    </div>

                    <div class="category-summary-card category-summary-blue">
                        <div class="category-summary-header">
                            <p class="category-summary-label">Most Improved</p>
                            <div class="category-summary-icon category-summary-icon-blue">
                                <i data-lucide="trending-up"></i>
                            </div>
                        </div>
                        <p class="category-summary-title">N/A</p>
                        <p class="category-summary-subtitle">No data available</p>
                    </div>

                    <div class="category-summary-card category-summary-amber">
                        <div class="category-summary-header">
                            <p class="category-summary-label">Needs Focus</p>
                            <div class="category-summary-icon category-summary-icon-amber">
                                <i data-lucide="target"></i>
                            </div>
                        </div>
                        <p class="category-summary-title">N/A</p>
                        <p class="category-summary-subtitle">No data available</p>
                    </div>

                    <div class="category-summary-card category-summary-purple">
                        <div class="category-summary-header">
                            <p class="category-summary-label">Total Coverage</p>
                            <div class="category-summary-icon category-summary-icon-purple">
                                <i data-lucide="bar-chart-3"></i>
                            </div>
                        </div>
                        <p class="category-summary-title">${totalMetricsCount} Metrics</p>
                        <p class="category-summary-subtitle">Across ${categories.length} categories</p>
                    </div>
                </div>
            `;

            // Initialize Lucide icons
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
            return;
        }

        const topPerformer = categoriesWithValues.reduce((max, cat) => cat.value > max.value ? cat : max, categoriesWithValues[0]);

        const needsFocus = categoriesWithValues.reduce((min, cat) => cat.value < min.value ? cat : min, categoriesWithValues[0]);

        // Find most improved - this would need historical data, for now show the second highest
        const sortedByValue = [...categoriesWithValues].sort((a, b) => b.value - a.value);
        const mostImproved = sortedByValue.length > 1 ? sortedByValue[1] : sortedByValue[0];

        summaryContainer.innerHTML = `
            <div class="category-summary-content">
                <div class="category-summary-card category-summary-emerald">
                    <div class="category-summary-header">
                        <p class="category-summary-label">Top Performer</p>
                        <div class="category-summary-icon category-summary-icon-emerald">
                            <i data-lucide="award"></i>
                        </div>
                    </div>
                    <p class="category-summary-title">${topPerformer.name}</p>
                    <p class="category-summary-subtitle">${topPerformer.value !== null && topPerformer.value !== undefined ? `${topPerformer.value}% score` : 'N/A'}</p>
                </div>

                <div class="category-summary-card category-summary-blue">
                    <div class="category-summary-header">
                        <p class="category-summary-label">Most Improved</p>
                        <div class="category-summary-icon category-summary-icon-blue">
                            <i data-lucide="trending-up"></i>
                        </div>
                    </div>
                    <p class="category-summary-title">${mostImproved.name}</p>
                    <p class="category-summary-subtitle">${mostImproved.value !== null && mostImproved.value !== undefined ? `${mostImproved.value}% score` : 'N/A'}</p>
                </div>

                <div class="category-summary-card category-summary-amber">
                    <div class="category-summary-header">
                        <p class="category-summary-label">Needs Focus</p>
                        <div class="category-summary-icon category-summary-icon-amber">
                            <i data-lucide="target"></i>
                        </div>
                    </div>
                    <p class="category-summary-title">${needsFocus.name}</p>
                    <p class="category-summary-subtitle">${needsFocus.value !== null && needsFocus.value !== undefined ? `${needsFocus.value}% score` : 'N/A'}</p>
                </div>

                <div class="category-summary-card category-summary-purple">
                    <div class="category-summary-header">
                        <p class="category-summary-label">Total Coverage</p>
                        <div class="category-summary-icon category-summary-icon-purple">
                            <i data-lucide="bar-chart-3"></i>
                        </div>
                    </div>
                    <p class="category-summary-title">${totalMetricsCount} Metrics</p>
                    <p class="category-summary-subtitle">Across ${categories.length} categories</p>
                </div>
            </div>
        `;

        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
}

async function renderLatencyDistributionChart() {
    const container = document.getElementById('latencyDistributionChart');
    const legendContainer = document.getElementById('latencyDistributionLegend');
    if (!container || typeof ApexCharts === 'undefined') return;

    try {
        // Fetch test results to calculate latency percentiles
        const response = await fetch('/api/tests?limit=1000');
        const tests = await response.json();

        // Extract durations and convert to ms
        const durations = tests
            .filter(t => t.duration && t.duration > 0)
            .map(t => t.duration * 1000) // Convert to ms
            .sort((a, b) => a - b);

        // Target values for each percentile (from dream dashboard)
        const targets = {
            P50: 1000,
            P75: 1200,
            P90: 1500,
            P95: 1800,
            P99: 2500
        };

        let data = [];
        let avgLatency = 0;

        if (durations.length > 0) {
            // Calculate percentiles
            const percentile = (arr, p) => {
                if (arr.length === 0) return 0;
                const index = Math.ceil((p / 100) * arr.length) - 1;
                return arr[Math.max(0, index)];
            };

            // Calculate average latency
            avgLatency = durations.reduce((a, b) => a + b, 0) / durations.length;

            // Define colors and labels for each percentile
            const percentileConfig = [
                { name: 'P50', color: '#10b981', label: 'Fast' },
                { name: 'P75', color: '#3b82f6', label: 'Good' },
                { name: 'P90', color: '#8b5cf6', label: 'Average' },
                { name: 'P95', color: '#f59e0b', label: 'Slow' },
                { name: 'P99', color: '#ef4444', label: 'Critical' }
            ];

            data = percentileConfig.map(config => ({
                name: config.name,
                value: Math.round(percentile(durations, parseInt(config.name.substring(1)))),
                color: config.color,
                label: config.label,
                target: targets[config.name]
            }));

            // Update average latency display
            const avgElement = document.getElementById('latencyAvgValue');
            if (avgElement) {
                if (avgLatency >= 1000) {
                    avgElement.textContent = `${(avgLatency / 1000).toFixed(2)}s`;
                } else {
                    avgElement.textContent = `${Math.round(avgLatency)}ms`;
                }
            }
        } else {
            // No data available
            container.innerHTML = '<div style="text-align: center; padding: 3rem; color: var(--text-secondary);">No latency data available yet. Run tests to see latency distribution.</div>';
            if (legendContainer) legendContainer.innerHTML = '';
            return;
        }

        if (charts.latencyDistribution) {
            charts.latencyDistribution.destroy();
        }

        // Store data for tooltip access
        window.latencyDistributionData = data;

        // Prepare data for line chart with smooth curve
        const percentileValues = data.map(d => d.value);
        const percentileNames = data.map(d => d.name);

        // Create a smooth gradient color for the line
        const lineColor = '#60a5fa';
        const gradientFrom = '#3b82f6';
        const gradientTo = '#8b5cf6';

        charts.latencyDistribution = new ApexCharts(container, {
            series: [{
                name: 'Latency',
                data: percentileValues
            }],
            chart: {
                type: 'line',
                height: 320,
                background: 'transparent',
                toolbar: { show: false },
                zoom: { enabled: false },
                fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 1500,
                    animateGradually: {
                        enabled: true,
                        delay: 150
                    }
                },
                dropShadow: {
                    enabled: true,
                    color: 'rgba(96, 165, 250, 0.4)',
                    blur: 15,
                    opacity: 0.6,
                    top: 10,
                    left: 0
                }
            },
            xaxis: {
                categories: percentileNames,
                labels: {
                    style: {
                        colors: '#94a3b8',
                        fontSize: '13px',
                        fontWeight: 600,
                        fontFamily: 'Inter, sans-serif'
                    }
                },
                axisBorder: {
                    show: true,
                    color: 'rgba(71, 85, 105, 0.5)',
                    strokeWidth: 1.5
                },
                axisTicks: {
                    show: false
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: '#94a3b8',
                        fontSize: '12px',
                        fontWeight: 500
                    },
                    formatter: (val) => {
                        // Always show in seconds if >= 1000ms, otherwise ms
                        if (val >= 1000) {
                            return `${(val / 1000).toFixed(1)}s`;
                        }
                        return `${Math.round(val)}ms`;
                    }
                },
                title: {
                    text: 'Latency',
                    style: {
                        color: '#94a3b8',
                        fontSize: '13px',
                        fontWeight: 600
                    },
                    offsetX: -10,
                    rotate: -90
                },
                axisBorder: {
                    show: true,
                    color: '#475569',
                    strokeWidth: 2
                },
                axisTicks: {
                    show: false
                },
                // Ensure proper scaling
                min: 0,
                forceNiceScale: true,
                decimalsInFloat: 1
            },
            colors: [lineColor],
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'dark',
                    type: 'vertical',
                    shadeIntensity: 0.6,
                    gradientToColors: [gradientTo],
                    inverseColors: false,
                    opacityFrom: 0.8,
                    opacityTo: 0.15,
                    stops: [0, 50, 100],
                    colorStops: [
                        {
                            offset: 0,
                            color: gradientFrom,
                            opacity: 0.8
                        },
                        {
                            offset: 50,
                            color: lineColor,
                            opacity: 0.5
                        },
                        {
                            offset: 100,
                            color: gradientTo,
                            opacity: 0.15
                        }
                    ]
                }
            },
            stroke: {
                curve: 'smooth',
                width: 3.5,
                colors: [lineColor],
                lineCap: 'round'
            },
            markers: {
                size: [6, 6, 6, 6, 6],
                colors: data.map(d => d.color),
                strokeColors: '#0f172a',
                strokeWidth: 2,
                hover: {
                    size: 8
                },
                shape: 'circle'
            },
            dataLabels: {
                enabled: false // Disable default labels, we'll use custom tooltip
            },
            tooltip: {
                theme: 'dark',
                enabled: true,
                shared: false,
                intersect: true,
                followCursor: true,
                custom: function({ series, seriesIndex, dataPointIndex, w }) {
                    const value = series[seriesIndex][dataPointIndex];
                    const percentileData = window.latencyDistributionData[dataPointIndex];
                    if (!percentileData) {
                        return `<div style="padding: 10px 14px; background: rgba(15, 23, 42, 0.98); border: 1px solid rgba(71, 85, 105, 0.6); border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">${value >= 1000 ? (value / 1000).toFixed(2) + 's' : value + 'ms'}</div>`;
                    }
                    const target = percentileData.target;
                    const performance = ((target - value) / target * 100).toFixed(1);
                    const isGood = performance > 0;
                    const valueFormatted = value >= 1000 ? (value / 1000).toFixed(2) + 's' : Math.round(value) + 'ms';
                    const targetFormatted = target >= 1000 ? (target / 1000).toFixed(1) + 's' : target + 'ms';

                    return `
                        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.98)); border: 1px solid rgba(71, 85, 105, 0.6); border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 30px ${percentileData.color}40; backdrop-filter: blur(20px); padding: 14px 18px; min-width: 200px;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                                <div style="width: 12px; height: 12px; border-radius: 50%; background: ${percentileData.color}; box-shadow: 0 0 12px ${percentileData.color}80;"></div>
                                <div style="color: #e2e8f0; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                                    ${percentileData.name} • ${percentileData.label}
                                </div>
                            </div>
                            <div style="color: ${percentileData.color}; font-weight: 700; font-size: 24px; margin-bottom: 6px; text-shadow: 0 0 20px ${percentileData.color}50;">
                                ${valueFormatted}
                            </div>
                            <div style="color: #94a3b8; font-size: 11px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                                <span>Target:</span>
                                <span style="color: #cbd5e1; font-weight: 500;">${targetFormatted}</span>
                            </div>
                            <div style="color: ${isGood ? '#10b981' : '#ef4444'}; font-size: 11px; font-weight: 600; display: flex; align-items: center; gap: 4px;">
                                <span>${isGood ? '✓' : '⚠'}</span>
                                <span>${Math.abs(performance)}% ${isGood ? 'under' : 'over'} target</span>
                            </div>
                        </div>
                    `;
                },
                style: {
                    fontSize: '12px',
                    fontFamily: 'Inter, sans-serif'
                }
            },
            grid: {
                borderColor: 'rgba(71, 85, 105, 0.3)',
                strokeDashArray: 4,
                xaxis: {
                    lines: {
                        show: false
                    }
                },
                yaxis: {
                    lines: {
                        show: true,
                        strokeDashArray: 4
                    }
                },
                padding: {
                    top: 25,
                    right: 15,
                    left: 15,
                    bottom: 10
                }
            },
            theme: { mode: 'dark' }
        });

        charts.latencyDistribution.render();

        // Render legend - matching dream dashboard exactly
        if (legendContainer) {
            const legendHtml = `
                <div class="latency-legend-content">
                    ${data.map((d) => {
                        const percentile = parseInt(d.name.substring(1));
                        const valueFormatted = d.value >= 1000 ? (d.value / 1000).toFixed(2) + 's' : d.value + 'ms';
                        // Get gradient colors for each indicator
                        const gradientColors = {
                            'P50': { from: '#10b981', to: '#059669' },
                            'P75': { from: '#3b82f6', to: '#2563eb' },
                            'P90': { from: '#8b5cf6', to: '#7c3aed' },
                            'P95': { from: '#f59e0b', to: '#d97706' },
                            'P99': { from: '#ef4444', to: '#dc2626' }
                        };
                        const gradient = gradientColors[d.name] || { from: d.color, to: d.color };
                        const labelText = d.label === 'Average' ? 'Avg' : d.label;
                        return `
                        <div class="latency-legend-item">
                            <div class="latency-legend-indicator" style="background: linear-gradient(to bottom right, ${gradient.from}, ${gradient.to}); box-shadow: 0 4px 12px ${d.color}30;"></div>
                            <div>
                                <p class="latency-legend-label">${d.name} • ${labelText}</p>
                                <p class="latency-legend-value">${percentile}% under ${valueFormatted}</p>
                            </div>
                        </div>
                        `;
                    }).join('')}
                </div>
            `;
            legendContainer.innerHTML = legendHtml;
        }
    } catch (error) {
        console.error('Failed to render latency distribution:', error);
        container.innerHTML = '<div style="text-align: center; padding: 3rem; color: var(--text-secondary);">Failed to load latency data.</div>';
        if (legendContainer) legendContainer.innerHTML = '';
    }
}

// Helper function to convert hex to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
}

// ===== TEST RESULTS =====
let selectedTestId = null;
let allTestResults = [];
let filteredTestResults = [];
let currentTestPage = 1;
let currentFilter = 'all'; // 'all', 'passed', 'failed', 'skipped'
const TESTS_PER_PAGE = 10;

async function loadTestResults() {
    try {
        const response = await fetch('/api/tests?limit=1000');
        allTestResults = await response.json();

        // Update stats
        const passed = allTestResults.filter(t => t.status === 'passed').length;
        const failed = allTestResults.filter(t => t.status === 'failed').length;
        const skipped = allTestResults.filter(t => t.status === 'skipped').length;

        updateElement('testResultsPassed', passed);
        updateElement('testResultsFailed', failed);
        updateElement('testResultsSkipped', skipped);
        updateElement('testResultsAll', allTestResults.length);

        // Apply current filter without resetting page number
        applyTestFilter(false);
    } catch (error) {
        console.error('Failed to load test results:', error);
        document.getElementById('testResultsList').innerHTML =
            '<div class="no-data">Failed to load test results. Please try again later.</div>';
        document.getElementById('testResultsPagination').innerHTML = '';
    }
}

function applyTestFilter(resetPage = true) {
    if (currentFilter === 'all') {
        filteredTestResults = allTestResults;
    } else {
        filteredTestResults = allTestResults.filter(t => t.status === currentFilter);
    }

    // Only reset to page 1 when filter changes, not on reload
    if (resetPage) {
        currentTestPage = 1;
    }

    // Update filter button states
    updateFilterButtons();

    // Render filtered results
    renderTestResults();
}

// Make setTestFilter globally accessible
window.setTestFilter = function(filter) {
    if (currentFilter === filter) {
        // Toggle off if clicking the same filter
        currentFilter = 'all';
    } else {
        currentFilter = filter;
    }
    applyTestFilter();
};

function updateFilterButtons() {
    const passedBtn = document.getElementById('filterPassed');
    const failedBtn = document.getElementById('filterFailed');
    const skippedBtn = document.getElementById('filterSkipped');
    const allBtn = document.getElementById('filterAll');

    if (passedBtn) {
        passedBtn.classList.toggle('active', currentFilter === 'passed');
    }
    if (failedBtn) {
        failedBtn.classList.toggle('active', currentFilter === 'failed');
    }
    if (skippedBtn) {
        skippedBtn.classList.toggle('active', currentFilter === 'skipped');
    }
    if (allBtn) {
        allBtn.classList.toggle('active', currentFilter === 'all');
    }
}

function renderTestResults() {
    const container = document.getElementById('testResultsList');
    if (!container) return;

    if (filteredTestResults.length === 0) {
        container.innerHTML = `<div class="no-data">No ${currentFilter === 'all' ? '' : currentFilter} test results available</div>`;
        renderPagination();
        return;
    }

    // Calculate pagination based on filtered results
    const totalPages = Math.ceil(filteredTestResults.length / TESTS_PER_PAGE);
    const startIndex = (currentTestPage - 1) * TESTS_PER_PAGE;
    const endIndex = startIndex + TESTS_PER_PAGE;
    const currentTests = filteredTestResults.slice(startIndex, endIndex);

    container.innerHTML = currentTests.map(test => {
        const timestamp = new Date(test.timestamp).toLocaleString();
        const duration = test.duration ? `${test.duration.toFixed(2)}s` : 'N/A';
        const isExpanded = selectedTestId === test.test_id;
        // Map status to card class
        const cardClass = test.status === 'failed' ? 'failed' :
                         test.status === 'skipped' ? 'skipped' : 'passed';

        // Only allow expansion for failed tests
        const canExpand = test.status === 'failed';
        const shouldShowExpanded = canExpand && isExpanded;

        return `
            <div class="test-result-card ${cardClass}" data-test-id="${test.test_id}">
                <div class="test-result-header" ${canExpand ? `onclick="toggleTestDetails('${test.test_id}')"` : ''} ${!canExpand ? 'style="cursor: default;"' : ''}>
                    <div class="test-result-main">
                        <div class="test-status-indicator ${test.status}"></div>
                        <div class="test-info">
                            <div class="test-name">${escapeHtml(test.test_name)}</div>
                            <div class="test-meta">
                                <span>${timestamp}</span>
                                <span class="meta-separator">•</span>
                                <span>Duration: ${duration}</span>
                                <span class="meta-separator">•</span>
                                <span>Type: ${test.test_type || 'general'}</span>
                            </div>
                        </div>
                    </div>
                    <div class="test-result-actions">
                        <span class="test-status-badge ${test.status}">${test.status.toUpperCase()}</span>
                        ${canExpand ? `<i data-lucide="chevron-down" class="test-expand-icon ${isExpanded ? 'expanded' : ''}"></i>` : ''}
                    </div>
                </div>
                ${shouldShowExpanded ? '<div class="test-details-loading">Loading details...</div>' : ''}
            </div>
        `;
    }).join('');

    // Re-initialize icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Load details for expanded tests (only for failed)
    if (selectedTestId) {
        const expandedTest = currentTests.find(t => t.test_id === selectedTestId);
        if (expandedTest && expandedTest.status === 'failed') {
            loadTestDetails(expandedTest);
        }
    }

    // Render pagination
    renderPagination();
}

function renderPagination() {
    const paginationContainer = document.getElementById('testResultsPagination');
    if (!paginationContainer) return;

    const totalPages = Math.ceil(filteredTestResults.length / TESTS_PER_PAGE);

    if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        return;
    }

    const startItem = (currentTestPage - 1) * TESTS_PER_PAGE + 1;
    const endItem = Math.min(currentTestPage * TESTS_PER_PAGE, filteredTestResults.length);

    let paginationHTML = `
        <button class="pagination-btn" onclick="goToTestPage(${currentTestPage - 1})" ${currentTestPage === 1 ? 'disabled' : ''}>
            <i data-lucide="chevron-left"></i>
        </button>
    `;

    // Show page numbers (max 5 visible)
    const maxVisible = 5;
    let startPage = Math.max(1, currentTestPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);

    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        paginationHTML += `<button class="pagination-btn" onclick="goToTestPage(1)">1</button>`;
        if (startPage > 2) {
            paginationHTML += `<span class="pagination-info">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        paginationHTML += `
            <button class="pagination-btn ${i === currentTestPage ? 'active' : ''}" onclick="goToTestPage(${i})">
                ${i}
            </button>
        `;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            paginationHTML += `<span class="pagination-info">...</span>`;
        }
        paginationHTML += `<button class="pagination-btn" onclick="goToTestPage(${totalPages})">${totalPages}</button>`;
    }

    paginationHTML += `
        <button class="pagination-btn" onclick="goToTestPage(${currentTestPage + 1})" ${currentTestPage === totalPages ? 'disabled' : ''}>
            <i data-lucide="chevron-right"></i>
        </button>
        <span class="pagination-info">Showing ${startItem}-${endItem} of ${filteredTestResults.length}</span>
    `;

    paginationContainer.innerHTML = paginationHTML;

    // Re-initialize icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

// Make goToTestPage globally accessible
window.goToTestPage = function(page) {
    const totalPages = Math.ceil(filteredTestResults.length / TESTS_PER_PAGE);
    if (page < 1 || page > totalPages) return;

    currentTestPage = page;
    renderTestResults();

    // Scroll to top of container
    const container = document.querySelector('.test-results-container');
    if (container) {
        // For internal container scrolling, use native scroll
        // Lenis handles main window scrolling
        container.scrollTo({ top: 0, behavior: 'auto' });
    }
};

async function loadTestDetails(test) {
    const card = document.querySelector(`[data-test-id="${test.test_id}"]`);
    if (!card) return;

    const detailsContainer = card.querySelector('.test-details-loading');
    if (!detailsContainer) return;

    // Check test status first
    if (test.status === 'passed') {
        detailsContainer.outerHTML = renderBasicTestDetails(test);
        if (typeof lucide !== 'undefined') lucide.createIcons();
        return;
    }

    // For skipped tests, show simple skip message
    if (test.status === 'skipped') {
        detailsContainer.outerHTML = renderSkippedTestDetails(test);
        if (typeof lucide !== 'undefined') lucide.createIcons();
        return;
    }

    // For failed tests, try to fetch detailed failure information
    try {
        const response = await fetch(`/api/tests/failed/${test.test_id}/details`);
        if (!response.ok) {
            // If details not available, show basic failure message
            detailsContainer.outerHTML = renderBasicFailureDetails(test);
            if (typeof lucide !== 'undefined') lucide.createIcons();
            return;
        }

        const details = await response.json();
        detailsContainer.outerHTML = renderFailedTestDetails(test, details);
        if (typeof lucide !== 'undefined') lucide.createIcons();
    } catch (error) {
        // On error, show basic failure message for failed tests
        detailsContainer.outerHTML = renderBasicFailureDetails(test);
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }
}

async function toggleTestDetails(testId) {
    // Find the test to check if it can be expanded
    const test = allTestResults.find(t => t.test_id === testId);
    if (!test) return;

    // Only allow expansion for failed tests
    if (test.status !== 'failed') {
        return; // Only failed tests can expand
    }

    if (selectedTestId === testId) {
        selectedTestId = null;
    } else {
        selectedTestId = testId;
    }

    // Re-render current page without resetting pagination
    renderTestResults();
}


// ===== GENERIC ASSERTION ERROR COMPONENT =====
// Simple, clean component for displaying assertion errors - shows assertion as-is without confusing parsing
function renderAssertionErrorComponent(parsedError, stackTrace = null) {
    if (!parsedError || (!parsedError.assertion && !parsedError.message)) {
        return '';
    }

    return `
        <div class="assertion-error-component">
            <!-- Error Header -->
            <div class="assertion-component-header">
                <div class="assertion-component-title-section">
                    <div class="assertion-component-icon-wrapper">
                        <i data-lucide="alert-octagon"></i>
                    </div>
                    <div class="assertion-component-title-group">
                        <div class="assertion-component-error-type">${escapeHtml(parsedError.errorType || 'AssertionError')}</div>
                        ${parsedError.message ? `
                            <div class="assertion-component-message">${escapeHtml(parsedError.message)}</div>
                        ` : ''}
                    </div>
                </div>
                ${parsedError.filePath ? `
                    <div class="assertion-component-location-badge">
                        <i data-lucide="map-pin"></i>
                        <span class="assertion-location-file">${escapeHtml(parsedError.filePath)}</span>
                        ${parsedError.lineNumber ? `<span class="assertion-location-line">:${parsedError.lineNumber}</span>` : ''}
                    </div>
                ` : ''}
            </div>

            <!-- Code Context - Shows the original code with variable names -->
            ${parsedError.codeSnippet ? `
                <div class="assertion-component-code-context">
                    <div class="assertion-component-section-label">
                        <i data-lucide="file-code"></i>
                        <span>Code Context</span>
                    </div>
                    <div class="assertion-component-code-block">
                        <pre><code>${escapeHtml(parsedError.codeSnippet)}</code></pre>
                    </div>
                </div>
            ` : ''}

            <!-- Stack Trace (Collapsible) -->
            ${stackTrace ? `
                <div class="assertion-component-stack-trace">
                    <details class="assertion-stack-details">
                        <summary class="assertion-stack-summary">
                            <div class="assertion-component-section-label">
                                <i data-lucide="layers"></i>
                                <span>Stack Trace</span>
                            </div>
                            <i data-lucide="chevron-down" class="stack-chevron"></i>
                        </summary>
                        <div class="assertion-stack-content">
                            <pre><code>${escapeHtml(stackTrace)}</code></pre>
                        </div>
                    </details>
                </div>
            ` : ''}
        </div>
    `;
}

// Parse assertion errors to extract structured information
function parseAssertionError(errorMessage, stackTrace) {
    const parsed = {
        errorType: 'Unknown Error',
        filePath: null,
        lineNumber: null,
        functionName: null,
        assertion: null,
        message: errorMessage,
        codeSnippet: null,
        fullTrace: stackTrace
    };

    if (!errorMessage && !stackTrace) return parsed;

    const fullText = stackTrace ? `${errorMessage}\n${stackTrace}` : errorMessage;

    // Extract error type (check both start and anywhere in message)
    const errorTypeMatch = errorMessage.match(/(\w+Error):/);
    if (errorTypeMatch) {
        parsed.errorType = errorTypeMatch[1];
    } else if (errorMessage.includes('AssertionError')) {
        parsed.errorType = 'AssertionError';
    }

    // Extract file path and line number from stack trace
    const fileMatch = fullText.match(/([^\s]+\.py):(\d+):\s+in\s+([^\s]+)/);
    if (fileMatch) {
        parsed.filePath = fileMatch[1];
        parsed.lineNumber = parseInt(fileMatch[2]);
        parsed.functionName = fileMatch[3];
    }

    // Extract assertion - just get the full assertion, don't try to parse expected/actual
    // This avoids confusion (e.g., "assert x > 0.1" doesn't mean expected is 0.1)
    const eLineMatch = fullText.match(/E\s+assert\s+([^\n]+)/);
    if (eLineMatch) {
        parsed.assertion = eLineMatch[1].trim();
    } else {
        // Fallback: try to find any assert statement
        const assertMatch = fullText.match(/assert\s+([^\n,]+)/);
        if (assertMatch) {
            parsed.assertion = assertMatch[1].trim();
        }
    }

    // Extract assertion message from code snippet (e.g., assert x, "message")
    const messageInCode = fullText.match(/assert\s+[^,]+,\s*"([^"]+)"/);
    if (messageInCode && !parsed.message) {
        parsed.message = messageInCode[1];
    }

    // Extract code snippet from stack trace
    const codeSnippetMatch = fullText.match(/([^\s]+\.py:\d+):\s+in\s+[^\n]+\n\s+([^\n]+)/);
    if (codeSnippetMatch) {
        parsed.codeSnippet = codeSnippetMatch[2].trim();
    }

    // Extract error message (text after AssertionError:)
    const messageMatch = errorMessage.match(/:\s*(.+)/);
    if (messageMatch) {
        parsed.message = messageMatch[1].trim();
    }

    return parsed;
}

function renderFailedTestDetails(test, details) {
    // Prioritize error message from test_info (pytest error), then failure_analysis
    const errorMessage = details.test_info?.error_message ||
                        details.failure_analysis?.root_cause ||
                        details.failure_analysis?.category ||
                        'Test validation failed';
    const stackTrace = details.test_info?.stack_trace || null;

    // Parse assertion error for better display
    const parsedError = parseAssertionError(errorMessage, stackTrace);

    // Extract metrics from validation scores and all_scores
    const metrics = {};

    // Add validation scores
    if (details.validation_scores) {
        Object.assign(metrics, details.validation_scores);
    }

    // Add scores from all_scores (convert 0-1 to percentage if needed)
    if (details.all_scores && Array.isArray(details.all_scores)) {
        details.all_scores.forEach(score => {
            let value = score.metric_value;
            // Convert 0-1 range to percentage for display
            if (typeof value === 'number' && value <= 1 && value >= 0) {
                value = value * 100;
            }
            metrics[score.metric_name] = value;
        });
    }

    // Filter and format metrics for display
    const metricEntries = Object.entries(metrics)
        .filter(([key]) => {
            const keyLower = key.toLowerCase();
            // Include similarity and is_relevant, but exclude threshold and value (blob data)
            return !['threshold', 'value'].includes(keyLower);
        })
        .slice(0, 6)
        .map(([key, value]) => {
            let displayValue = value;
            // Convert 0-1 range to percentage for display (for similarity, is_relevant, etc.)
            if (typeof value === 'number' && value <= 1 && value >= 0) {
                // For boolean metrics like is_relevant (0.0 or 1.0), show as percentage
                // For similarity scores (0-1), also show as percentage
                displayValue = (value * 100).toFixed(1);
            } else if (typeof value === 'number') {
                displayValue = value.toFixed(1);
            } else {
                displayValue = String(value);
            }
            return {
                name: formatMetricName(key),
                value: displayValue
            };
        });

    // Get expected and actual responses, with better handling
    const expected = details.expected_response || null;
    const actual = details.actual_response || null;
    const prompt = details.prompt || null;

    // Generate comparison HTML
    const comparison = highlightDifferences(expected, actual, prompt);

    return `
        <div class="test-details">
            <!-- Generic Assertion Error Component -->
            ${renderAssertionErrorComponent(parsedError, stackTrace)}

            ${metricEntries.length > 0 ? `
            <div class="test-metrics-section">
                <h4 class="test-section-title">AI Quality Metrics</h4>
                <div class="test-metrics-grid">
                    ${metricEntries.map(metric => `
                        <div class="test-metric-card">
                            <div class="test-metric-label">${escapeHtml(metric.name)}</div>
                            <div class="test-metric-value ${getMetricStatusClass(metric.value)}">${metric.value}%</div>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <div class="test-comparison-section">
                <h4 class="test-section-title">
                    <i data-lucide="eye"></i>
                    Side-by-Side Comparison
                </h4>
                ${prompt ? `
                <div class="test-prompt-section">
                    <div class="test-prompt-label">Query/Prompt:</div>
                    <div class="test-prompt-content">${escapeHtml(prompt)}</div>
                </div>
                ` : ''}
                <div class="test-comparison-grid">
                    <div class="test-comparison-card expected">
                        <div class="test-comparison-header">
                            <div class="test-comparison-indicator"></div>
                            <span>EXPECTED OUTPUT</span>
                        </div>
                        <div class="test-comparison-content">${comparison.expected}</div>
                    </div>
                    <div class="test-comparison-card actual">
                        <div class="test-comparison-header">
                            <div class="test-comparison-indicator"></div>
                            <span>ACTUAL OUTPUT</span>
                        </div>
                        <div class="test-comparison-content">${comparison.actual}</div>
                    </div>
                </div>
                <div class="test-comparison-info">
                    <i data-lucide="info"></i>
                    <div>
                        <div class="test-info-title">Difference Highlighting</div>
                        <div class="test-info-text">
                            <span class="highlight-expected">Pink highlights</span> show expected content •
                            <span class="highlight-actual">Yellow highlights</span> show actual output differences
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderBasicTestDetails(test) {
    // Only show success message for tests that actually passed
    if (test.status !== 'passed') {
        return renderBasicFailureDetails(test);
    }

    // For passed tests, don't show redundant message - status is already clear from badge/indicator
    return `
        <div class="test-details">
            <!-- Passed tests don't need additional details - status is already clear -->
        </div>
    `;
}

function renderSkippedTestDetails(test) {
    const skipReason = test.error_message || test.skip_reason || 'Test was skipped during execution.';

    return `
        <div class="test-details">
            <div class="skipped-test-info">
                <div class="skipped-test-icon">
                    <i data-lucide="skip-forward"></i>
                </div>
                <div class="skipped-test-content">
                    <div class="skipped-test-title">Test Skipped</div>
                    <div class="skipped-test-message">${escapeHtml(skipReason)}</div>
                </div>
            </div>
        </div>
    `;
}

function renderBasicFailureDetails(test) {
    // Try to get error message from test object if available
    const errorMessage = test.error_message ||
                        'This test failed during execution. Detailed failure information is not available.';
    const stackTrace = test.stack_trace || null;

    // Parse assertion error for better display
    const parsedError = parseAssertionError(errorMessage, stackTrace);

    return `
        <div class="test-details">
            <!-- Generic Assertion Error Component -->
            ${renderAssertionErrorComponent(parsedError, stackTrace)}
        </div>
    `;
}

function highlightDifferences(expected, actual, prompt = null) {
    // Handle null/undefined/empty values
    if (!expected && !actual) {
        return {
            expected: '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>',
            actual: '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>'
        };
    }

    if (!expected) {
        return {
            expected: '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>',
            actual: actual ? escapeHtml(actual) : '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>'
        };
    }

    if (!actual) {
        return {
            expected: expected ? escapeHtml(expected) : '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>',
            actual: '<span style="color: var(--text-secondary); font-style: italic;">N/A</span>'
        };
    }

    // Check if expected is a description (starts with "Expected:" or similar)
    const isDescription = expected.startsWith('Expected:') ||
                         expected.startsWith('Expected response to address:');

    if (isDescription) {
        // For descriptions, show the description clearly and the actual output
        let descriptionText = expected;
        if (expected.startsWith('Expected:')) {
            descriptionText = expected.replace('Expected:', '<strong>Expected:</strong>');
        } else if (expected.startsWith('Expected response to address:')) {
            // Legacy format - just show as description without extracting query
            descriptionText = expected.replace('Expected response to address:', '<strong>Expected:</strong> Response should address');
        }

        return {
            expected: descriptionText,
            actual: escapeHtml(actual)
        };
    }

    // For actual comparisons, use a smarter diff algorithm
    // Split into words, but preserve punctuation
    const expectedWords = expected.match(/\S+/g) || [];
    const actualWords = actual.match(/\S+/g) || [];

    // Use longest common subsequence for better matching
    const lcs = computeLCS(expectedWords, actualWords);
    const expectedSet = new Set(lcs);
    const actualSet = new Set(lcs);

    let expectedHtml = '';
    let actualHtml = '';

    // Build expected HTML with highlights
    expectedWords.forEach((word, i) => {
        if (i > 0) expectedHtml += ' ';
        if (expectedSet.has(word) && actualWords.includes(word)) {
            // Word exists in both - no highlight
            expectedHtml += escapeHtml(word);
        } else {
            // Word only in expected - highlight
            expectedHtml += `<span class="diff-highlight expected">${escapeHtml(word)}</span>`;
        }
    });

    // Build actual HTML with highlights
    actualWords.forEach((word, i) => {
        if (i > 0) actualHtml += ' ';
        if (actualSet.has(word) && expectedWords.includes(word)) {
            // Word exists in both - no highlight
            actualHtml += escapeHtml(word);
        } else {
            // Word only in actual - highlight
            actualHtml += `<span class="diff-highlight actual">${escapeHtml(word)}</span>`;
        }
    });

    return { expected: expectedHtml, actual: actualHtml };
}

// Helper function to compute Longest Common Subsequence
function computeLCS(arr1, arr2) {
    const m = arr1.length;
    const n = arr2.length;
    const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    // Build LCS table
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (arr1[i - 1] === arr2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // Reconstruct LCS
    const lcs = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
        if (arr1[i - 1] === arr2[j - 1]) {
            lcs.unshift(arr1[i - 1]);
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }

    return lcs;
}

function formatMetricName(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .trim()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

function getMetricStatusClass(value) {
    const numValue = typeof value === 'number' ? value : parseFloat(value);
    if (numValue < 70) return 'metric-low';
    if (numValue < 85) return 'metric-medium';
    return 'metric-high';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== EXPORT FUNCTIONALITY =====
function exportAllMetrics() {
    const allMetrics = {
        timestamp: new Date().toISOString(),
        timeRange: getTimeRangeValue(),
        criticalMetrics: [],
        baseModel: metricsData.baseModel,
        rag: metricsData.rag,
        safety: metricsData.safety,
        performance: metricsData.performance,
        reliability: metricsData.reliability,
        agent: metricsData.agent,
        security: metricsData.security
    };

    const json = JSON.stringify(allMetrics, null, 2);

    // Copy to clipboard
    if (navigator.clipboard) {
        navigator.clipboard.writeText(json).then(() => {
            alert('✓ All metrics copied to clipboard as JSON');
        }).catch(err => {
            console.error('Failed to copy:', err);
            fallbackCopyToClipboard(json);
        });
    } else {
        fallbackCopyToClipboard(json);
    }
}

function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    try {
        document.execCommand('copy');
        alert('✓ All metrics copied to clipboard as JSON');
    } catch (err) {
        console.error('Failed to copy:', err);
        alert('Failed to copy metrics. Please check console for JSON data.');
    }
    document.body.removeChild(textArea);
}

// ===== WEBSOCKET UPDATE HANDLER =====
function updateMetricsFromWebSocket(data) {
    if (data && data.tests) {
        // Update metrics when WebSocket sends new data
        loadAllMetrics();
    }
}
