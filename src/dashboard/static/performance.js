/**
 * PyAI-Slayer Dashboard - Performance & Optimization (Phase 6)
 * Virtual scrolling, lazy loading, PWA support, caching, memory optimization
 */

// ===== VIRTUAL SCROLLING =====
class VirtualScroller {
    constructor(container, itemHeight, renderItem) {
        this.container = container;
        this.itemHeight = itemHeight;
        this.renderItem = renderItem;
        this.data = [];
        this.scrollTop = 0;
        this.containerHeight = 0;
        this.init();
    }

    init() {
        this.container.style.overflowY = 'auto';
        this.container.style.position = 'relative';
        this.viewport = document.createElement('div');
        this.viewport.style.position = 'relative';
        this.container.appendChild(this.viewport);

        // Ultra-high-performance RAF for 120 FPS scrolling
        this.rafId = null;
        this.lastRenderTime = 0;
        this.scrollHandler = () => {
            const now = performance.now();
            // Throttle to max 120 FPS (8.33ms per frame)
            if (now - this.lastRenderTime >= 8.33) {
                if (this.rafId === null) {
                    this.rafId = requestAnimationFrame(() => {
                        this.onScroll();
                        this.lastRenderTime = performance.now();
                        this.rafId = null;
                    });
                }
            }
        };
        this.container.addEventListener('scroll', this.scrollHandler, { passive: true });
        this.updateContainerHeight();
    }

    updateContainerHeight() {
        this.containerHeight = this.container.clientHeight;
    }

    setData(data) {
        this.data = data;
        this.render();
    }

    onScroll() {
        this.scrollTop = this.container.scrollTop;
        this.render();
    }

    render() {
        const totalHeight = this.data.length * this.itemHeight;
        this.viewport.style.height = `${totalHeight}px`;

        const startIndex = Math.floor(this.scrollTop / this.itemHeight);
        const endIndex = Math.ceil((this.scrollTop + this.containerHeight) / this.itemHeight);

        // Add buffer for smoother scrolling
        const bufferSize = 5;
        const renderStart = Math.max(0, startIndex - bufferSize);
        const renderEnd = Math.min(this.data.length, endIndex + bufferSize);

        // Only re-render if visible range changed
        if (this.lastRenderStart === renderStart && this.lastRenderEnd === renderEnd) {
            return;
        }
        this.lastRenderStart = renderStart;
        this.lastRenderEnd = renderEnd;

        const fragment = document.createDocumentFragment();

        for (let i = renderStart; i < renderEnd; i++) {
            const item = this.renderItem(this.data[i], i);
            item.style.position = 'absolute';
            item.style.top = `${i * this.itemHeight}px`;
            item.style.width = '100%';
            fragment.appendChild(item);
        }

        this.viewport.innerHTML = '';
        this.viewport.appendChild(fragment);
    }
}

// ===== DEBOUNCE & THROTTLE =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ===== LAZY LOADING =====
class LazyLoader {
    constructor(options = {}) {
        this.options = {
            threshold: 0.1,
            rootMargin: '50px',
            ...options
        };
        this.observer = null;
        this.init();
    }

    init() {
        if ('IntersectionObserver' in window) {
            this.observer = new IntersectionObserver(
                this.handleIntersection.bind(this),
                this.options
            );
        }
    }

    observe(elements) {
        if (!this.observer) return;
        elements.forEach(el => this.observer.observe(el));
    }

    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;

                // Lazy load images
                if (target.dataset.src) {
                    target.src = target.dataset.src;
                    target.removeAttribute('data-src');
                }

                // Lazy load components
                if (target.dataset.component) {
                    this.loadComponent(target);
                }

                this.observer.unobserve(target);
            }
        });
    }

    loadComponent(element) {
        const componentName = element.dataset.component;
        // Component loading logic here
        element.classList.add('loaded');
    }
}

// ===== CACHING STRATEGIES =====
class CacheManager {
    constructor() {
        this.memoryCache = new Map();
        this.maxCacheSize = 100;
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
    }

    set(key, value, ttl = this.cacheExpiry) {
        // Implement LRU cache
        if (this.memoryCache.size >= this.maxCacheSize) {
            const firstKey = this.memoryCache.keys().next().value;
            this.memoryCache.delete(firstKey);
        }

        this.memoryCache.set(key, {
            value,
            expiry: Date.now() + ttl
        });
    }

    get(key) {
        const cached = this.memoryCache.get(key);

        if (!cached) return null;

        if (Date.now() > cached.expiry) {
            this.memoryCache.delete(key);
            return null;
        }

        return cached.value;
    }

    clear() {
        this.memoryCache.clear();
    }

    has(key) {
        return this.memoryCache.has(key) && Date.now() <= this.memoryCache.get(key).expiry;
    }
}

const cacheManager = new CacheManager();

// ===== API REQUEST OPTIMIZATION =====
class APIOptimizer {
    constructor() {
        this.pendingRequests = new Map();
        this.requestQueue = [];
        this.maxConcurrent = 6;
        this.activeRequests = 0;
    }

    async fetch(url, options = {}) {
        const cacheKey = `${url}_${JSON.stringify(options)}`;

        // Check cache first
        if (cacheManager.has(cacheKey)) {
            return cacheManager.get(cacheKey);
        }

        // Deduplicate concurrent identical requests
        if (this.pendingRequests.has(cacheKey)) {
            return this.pendingRequests.get(cacheKey);
        }

        const requestPromise = this.executeRequest(url, options, cacheKey);
        this.pendingRequests.set(cacheKey, requestPromise);

        try {
            const result = await requestPromise;
            cacheManager.set(cacheKey, result);
            return result;
        } finally {
            this.pendingRequests.delete(cacheKey);
        }
    }

    async executeRequest(url, options, cacheKey) {
        while (this.activeRequests >= this.maxConcurrent) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        this.activeRequests++;

        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } finally {
            this.activeRequests--;
        }
    }
}

const apiOptimizer = new APIOptimizer();

// ===== MEMORY OPTIMIZATION =====
class MemoryOptimizer {
    constructor() {
        this.weakMaps = new WeakMap();
        this.cleanupInterval = 60000; // 1 minute
        this.startCleanup();
    }

    startCleanup() {
        setInterval(() => {
            this.cleanup();
        }, this.cleanupInterval);
    }

    cleanup() {
        // Clear old cache entries
        cacheManager.memoryCache.forEach((value, key) => {
            if (Date.now() > value.expiry) {
                cacheManager.memoryCache.delete(key);
            }
        });

        // Clear old notifications (keep last 50)
        const notifications = JSON.parse(localStorage.getItem('dashboard-notifications') || '[]');
        if (notifications.length > 50) {
            localStorage.setItem('dashboard-notifications', JSON.stringify(notifications.slice(0, 50)));
        }

        // Clear old activity logs (keep last 100)
        const activities = JSON.parse(localStorage.getItem('dashboard-activity-log') || '[]');
        if (activities.length > 100) {
            localStorage.setItem('dashboard-activity-log', JSON.stringify(activities.slice(0, 100)));
        }

        // Memory cleanup completed
    }

    measureMemory() {
        if (performance.memory) {
            return {
                used: (performance.memory.usedJSHeapSize / 1048576).toFixed(2) + ' MB',
                total: (performance.memory.totalJSHeapSize / 1048576).toFixed(2) + ' MB',
                limit: (performance.memory.jsHeapSizeLimit / 1048576).toFixed(2) + ' MB'
            };
        }
        return null;
    }
}

const memoryOptimizer = new MemoryOptimizer();

// ===== PWA SERVICE WORKER =====
function registerServiceWorker() {
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    // Check for updates
                    registration.addEventListener('updatefound', () => {
                        const newWorker = registration.installing;
                        newWorker.addEventListener('statechange', () => {
                            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                showToast('Update Available', 'Refresh to get the latest version', 'info');
                            }
                        });
                    });
                })
                .catch(() => {
                    // ServiceWorker registration failed
                });
        });
    }
}

// ===== PERFORMANCE MONITORING =====
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            pageLoad: 0,
            apiCalls: [],
            renderTimes: [],
            memoryUsage: []
        };
        this.init();
    }

    init() {
        this.measurePageLoad();
        this.startMonitoring();
    }

    measurePageLoad() {
        window.addEventListener('load', () => {
            const perfData = performance.getEntriesByType('navigation')[0];
            if (perfData) {
                this.metrics.pageLoad = perfData.loadEventEnd - perfData.fetchStart;
            }
        });
    }

    startMonitoring() {
        // Monitor long tasks
        if ('PerformanceObserver' in window) {
            try {
                const observer = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.duration > 50) {
                            // Long task detected
                        }
                    }
                });
                observer.observe({ entryTypes: ['longtask'] });
            } catch (e) {
                // longtask not supported
            }
        }

        // Monitor memory periodically
        setInterval(() => {
            const memory = memoryOptimizer.measureMemory();
            if (memory) {
                this.metrics.memoryUsage.push({
                    timestamp: Date.now(),
                    ...memory
                });

                // Keep only last 10 measurements
                if (this.metrics.memoryUsage.length > 10) {
                    this.metrics.memoryUsage.shift();
                }
            }
        }, 30000); // Every 30 seconds
    }

    measureAPICall(name, duration) {
        this.metrics.apiCalls.push({
            name,
            duration,
            timestamp: Date.now()
        });

        if (this.metrics.apiCalls.length > 50) {
            this.metrics.apiCalls.shift();
        }
    }

    measureRenderTime(component, duration) {
        this.metrics.renderTimes.push({
            component,
            duration,
            timestamp: Date.now()
        });

        if (this.metrics.renderTimes.length > 50) {
            this.metrics.renderTimes.shift();
        }
    }

    getReport() {
        return {
            pageLoad: this.metrics.pageLoad,
            avgAPICall: this.metrics.apiCalls.length > 0
                ? this.metrics.apiCalls.reduce((sum, call) => sum + call.duration, 0) / this.metrics.apiCalls.length
                : 0,
            avgRenderTime: this.metrics.renderTimes.length > 0
                ? this.metrics.renderTimes.reduce((sum, rt) => sum + rt.duration, 0) / this.metrics.renderTimes.length
                : 0,
            currentMemory: this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1],
            cacheSize: cacheManager.memoryCache.size
        };
    }
}

const performanceMonitor = new PerformanceMonitor();

// ===== IMAGE OPTIMIZATION =====
function optimizeImages() {
    const images = document.querySelectorAll('img[data-src]');
    const lazyLoader = new LazyLoader();
    lazyLoader.observe(images);
}

// ===== BUNDLING & CODE SPLITTING =====
async function loadModule(moduleName) {
    const startTime = performance.now();

    try {
        const module = await import(`/static/${moduleName}.js`);
        const duration = performance.now() - startTime;
        performanceMonitor.measureAPICall(`load_${moduleName}`, duration);
        return module;
    } catch (error) {
        console.error(`Failed to load module ${moduleName}:`, error);
        return null;
    }
}

// ===== PREFETCHING =====
function prefetchResources(urls) {
    urls.forEach(url => {
        const link = document.createElement('link');
        link.rel = 'prefetch';
        link.href = url;
        document.head.appendChild(link);
    });
}

// ===== OPTIMIZED DATA LOADING =====
const optimizedLoadTests = debounce(async function() {
    const startTime = performance.now();

    try {
        const response = await apiOptimizer.fetch('/api/tests?hours=24');
        const duration = performance.now() - startTime;
        performanceMonitor.measureAPICall('load_tests', duration);

        // Process data efficiently
        return response;
    } catch (error) {
        console.error('Failed to load tests:', error);
        return [];
    }
}, 300);

// ===== LENIS INTEGRATION =====
// Helper function to check if Lenis is available
function isLenisAvailable() {
    return typeof window !== 'undefined' && window.lenis;
}

// Integrate VirtualScroller with Lenis for optimal performance
// Note: Lenis handles main window scrolling, VirtualScroller handles container scrolling
// They work together seamlessly

// ===== INITIALIZATION =====
function initPerformanceOptimizations() {
    // Register service worker for PWA
    registerServiceWorker();

    // Optimize images
    optimizeImages();

    // Prefetch common resources
    prefetchResources([
        '/api/statistics',
        '/api/tests/failed/patterns'
    ]);
}

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPerformanceOptimizations);
} else {
    initPerformanceOptimizations();
}

// Export for use in other modules
window.PerformanceUtils = {
    VirtualScroller,
    debounce,
    throttle,
    LazyLoader,
    cacheManager,
    apiOptimizer,
    memoryOptimizer,
    performanceMonitor,
    optimizedLoadTests
};
