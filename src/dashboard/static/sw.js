/**
 * PyAI-Slayer Dashboard - Service Worker
 * PWA offline support and caching
 */

const CACHE_NAME = 'pyai-slayer-v1';
const urlsToCache = [
    '/',
    '/static/styles.css',
    '/static/dashboard.js',
    '/static/charts_advanced.js',
    '/static/smart_features.js',
    '/static/export_reporting.js',
    '/static/collaboration.js',
    '/static/performance.js'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - network first, fallback to cache
self.addEventListener('fetch', (event) => {
    // API requests - network first
    if (event.request.url.includes('/api/')) {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Cache successful API responses
                    if (response && response.status === 200) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Fallback to cache if network fails
                    return caches.match(event.request);
                })
        );
        return;
    }

    // Static resources - cache first
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    return response;
                }

                return fetch(event.request).then((response) => {
                    // Don't cache non-successful responses
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }

                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });

                    return response;
                });
            })
    );
});

// Background sync
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-data') {
        event.waitUntil(syncData());
    }
});

async function syncData() {
    // Sync logic for offline data
}

// Push notifications
self.addEventListener('push', (event) => {
    const data = event.data ? event.data.json() : {};
    const title = data.title || 'PyAI-Slayer';
    const options = {
        body: data.body || 'New notification',
        icon: '/static/icon-192.png',
        badge: '/static/badge-72.png',
        data: data.url
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

// Notification click
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.notification.data) {
        event.waitUntil(
            clients.openWindow(event.notification.data)
        );
    }
});
