/**
 * PyAI-Slayer Dashboard - Collaboration & Notifications (Phase 5)
 * Alert rules, notification center, webhooks, test assignment, activity timeline
 */

// ===== NOTIFICATION CENTER =====
let notifications = JSON.parse(localStorage.getItem('dashboard-notifications')) || [];
let unreadCount = 0;

function initNotificationCenter() {
    updateNotificationBadge();
    loadNotifications();
}

function addNotification(type, title, message, metadata = {}) {
    const notification = {
        id: Date.now(),
        type, // 'test_failed', 'test_passed', 'alert_triggered', 'system'
        title,
        message,
        metadata,
        timestamp: new Date().toISOString(),
        read: false
    };

    notifications.unshift(notification);

    // Keep only last 100 notifications
    if (notifications.length > 100) {
        notifications = notifications.slice(0, 100);
    }

    localStorage.setItem('dashboard-notifications', JSON.stringify(notifications));
    updateNotificationBadge();

    // Show toast for immediate feedback
    showToast(title, message, type === 'test_failed' ? 'error' : type === 'alert_triggered' ? 'warning' : 'info');
}

function loadNotifications() {
    const container = document.getElementById('notificationList');
    if (!container) return;

    if (notifications.length === 0) {
        container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No notifications yet</div>';
        return;
    }

    container.innerHTML = notifications.map(notif => `
        <div class="notification-item ${notif.read ? 'read' : 'unread'}" onclick="markNotificationRead('${notif.id}')">
            <div class="notification-icon ${notif.type}">
                ${getNotificationIcon(notif.type)}
            </div>
            <div class="notification-content">
                <div class="notification-title">${notif.title}</div>
                <div class="notification-message">${notif.message}</div>
                <div class="notification-time">${formatTimeAgo(notif.timestamp)}</div>
            </div>
            ${!notif.read ? '<div class="notification-dot"></div>' : ''}
        </div>
    `).join('');
}

function markNotificationRead(id) {
    const notif = notifications.find(n => n.id == id);
    if (notif && !notif.read) {
        notif.read = true;
        localStorage.setItem('dashboard-notifications', JSON.stringify(notifications));
        updateNotificationBadge();
        loadNotifications();
    }
}

function markAllRead() {
    notifications.forEach(n => n.read = true);
    localStorage.setItem('dashboard-notifications', JSON.stringify(notifications));
    updateNotificationBadge();
    loadNotifications();
    showToast('All Read', 'All notifications marked as read', 'success');
}

function clearAllNotifications() {
    if (!confirm('Clear all notifications?')) return;
    notifications = [];
    localStorage.setItem('dashboard-notifications', JSON.stringify(notifications));
    updateNotificationBadge();
    loadNotifications();
    showToast('Cleared', 'All notifications cleared', 'success');
}

function updateNotificationBadge() {
    unreadCount = notifications.filter(n => !n.read).length;
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        badge.textContent = unreadCount;
        badge.style.display = unreadCount > 0 ? 'block' : 'none';
    }
}

function toggleNotificationCenter() {
    const panel = document.getElementById('notificationPanel');
    if (panel) {
        const isVisible = panel.style.display === 'block';
        panel.style.display = isVisible ? 'none' : 'block';
        if (!isVisible) {
            loadNotifications();
        }
    }
}

function getNotificationIcon(type) {
    const icons = {
        test_failed: '❌',
        test_passed: '✅',
        alert_triggered: '⚠️',
        system: 'ℹ️',
        webhook: '🔗',
        assignment: '👤'
    };
    return icons[type] || 'ℹ️';
}

function formatTimeAgo(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = Math.floor((now - time) / 1000); // seconds

    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// ===== ALERT RULES =====
let alertRules = JSON.parse(localStorage.getItem('dashboard-alert-rules')) || [];

function createAlertRule(rule) {
    const newRule = {
        id: Date.now(),
        name: rule.name,
        enabled: true,
        condition: rule.condition, // 'pass_rate_below', 'failure_count_above', 'duration_above'
        threshold: rule.threshold,
        channel: rule.channel, // 'notification', 'webhook', 'email'
        webhookUrl: rule.webhookUrl || '',
        createdAt: new Date().toISOString()
    };

    alertRules.push(newRule);
    localStorage.setItem('dashboard-alert-rules', JSON.stringify(alertRules));
    loadAlertRules();
    showToast('Alert Created', `Rule "${rule.name}" created successfully`, 'success');
}

function loadAlertRules() {
    const container = document.getElementById('alertRulesList');
    if (!container) return;

    if (alertRules.length === 0) {
        container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No alert rules configured</div>';
        return;
    }

    container.innerHTML = alertRules.map(rule => `
        <div class="alert-rule-item">
            <div class="alert-rule-header">
                <div>
                    <strong>${rule.name}</strong>
                    <span class="alert-rule-status ${rule.enabled ? 'enabled' : 'disabled'}">
                        ${rule.enabled ? '● Active' : '○ Disabled'}
                    </span>
                </div>
                <div class="alert-rule-actions">
                    <button onclick="toggleAlertRule(${rule.id})" class="secondary" style="padding: 0.5rem 1rem;">
                        ${rule.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button onclick="deleteAlertRule(${rule.id})" class="danger" style="padding: 0.5rem 1rem;">
                        Delete
                    </button>
                </div>
            </div>
            <div class="alert-rule-details">
                <span>Condition: <strong>${formatCondition(rule.condition)}</strong></span>
                <span>Threshold: <strong>${rule.threshold}${getThresholdUnit(rule.condition)}</strong></span>
                <span>Channel: <strong>${rule.channel}</strong></span>
            </div>
        </div>
    `).join('');
}

function toggleAlertRule(id) {
    const rule = alertRules.find(r => r.id === id);
    if (rule) {
        rule.enabled = !rule.enabled;
        localStorage.setItem('dashboard-alert-rules', JSON.stringify(alertRules));
        loadAlertRules();
        showToast('Alert Updated', `Rule ${rule.enabled ? 'enabled' : 'disabled'}`, 'info');
    }
}

function deleteAlertRule(id) {
    if (!confirm('Delete this alert rule?')) return;
    alertRules = alertRules.filter(r => r.id !== id);
    localStorage.setItem('dashboard-alert-rules', JSON.stringify(alertRules));
    loadAlertRules();
    showToast('Alert Deleted', 'Rule removed successfully', 'success');
}

function formatCondition(condition) {
    const map = {
        pass_rate_below: 'Pass Rate Below',
        failure_count_above: 'Failure Count Above',
        duration_above: 'Duration Above'
    };
    return map[condition] || condition;
}

function getThresholdUnit(condition) {
    if (condition === 'pass_rate_below') return '%';
    if (condition === 'duration_above') return 's';
    return '';
}

function checkAlertRules(stats) {
    alertRules.filter(rule => rule.enabled).forEach(rule => {
        let triggered = false;

        switch (rule.condition) {
            case 'pass_rate_below':
                triggered = (stats.pass_rate || 0) < rule.threshold;
                break;
            case 'failure_count_above':
                triggered = (stats.failed || 0) > rule.threshold;
                break;
            case 'duration_above':
                triggered = (stats.avg_duration || 0) > rule.threshold;
                break;
        }

        if (triggered) {
            handleAlertTrigger(rule, stats);
        }
    });
}

function handleAlertTrigger(rule, stats) {
    const message = `${rule.name}: ${formatCondition(rule.condition)} threshold (${rule.threshold}) exceeded`;

    addNotification('alert_triggered', 'Alert Triggered', message, { rule, stats });

    if (rule.channel === 'webhook' && rule.webhookUrl) {
        sendWebhook(rule.webhookUrl, {
            alert: rule.name,
            condition: rule.condition,
            threshold: rule.threshold,
            current_value: getCurrentValue(rule.condition, stats),
            timestamp: new Date().toISOString()
        });
    }
}

function getCurrentValue(condition, stats) {
    if (condition === 'pass_rate_below') return stats.pass_rate;
    if (condition === 'failure_count_above') return stats.failed;
    if (condition === 'duration_above') return stats.avg_duration;
    return null;
}

// ===== WEBHOOKS =====
let webhooks = JSON.parse(localStorage.getItem('dashboard-webhooks')) || [];

function addWebhook(webhook) {
    const newWebhook = {
        id: Date.now(),
        name: webhook.name,
        url: webhook.url,
        events: webhook.events || ['test_failed', 'alert_triggered'],
        enabled: true,
        createdAt: new Date().toISOString()
    };

    webhooks.push(newWebhook);
    localStorage.setItem('dashboard-webhooks', JSON.stringify(webhooks));
    loadWebhooks();
    showToast('Webhook Added', `"${webhook.name}" configured successfully`, 'success');
}

function loadWebhooks() {
    const container = document.getElementById('webhooksList');
    if (!container) return;

    if (webhooks.length === 0) {
        container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No webhooks configured</div>';
        return;
    }

    container.innerHTML = webhooks.map(webhook => `
        <div class="webhook-item">
            <div class="webhook-header">
                <div>
                    <strong>${webhook.name}</strong>
                    <span class="webhook-status ${webhook.enabled ? 'enabled' : 'disabled'}">
                        ${webhook.enabled ? '● Active' : '○ Disabled'}
                    </span>
                </div>
                <div>
                    <button onclick="toggleWebhook(${webhook.id})" class="secondary" style="padding: 0.5rem 1rem;">
                        ${webhook.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button onclick="testWebhook(${webhook.id})" style="padding: 0.5rem 1rem;">
                        Test
                    </button>
                    <button onclick="deleteWebhook(${webhook.id})" class="danger" style="padding: 0.5rem 1rem;">
                        Delete
                    </button>
                </div>
            </div>
            <div class="webhook-url">${webhook.url}</div>
            <div class="webhook-events">Events: ${webhook.events.join(', ')}</div>
        </div>
    `).join('');
}

function toggleWebhook(id) {
    const webhook = webhooks.find(w => w.id === id);
    if (webhook) {
        webhook.enabled = !webhook.enabled;
        localStorage.setItem('dashboard-webhooks', JSON.stringify(webhooks));
        loadWebhooks();
    }
}

function deleteWebhook(id) {
    if (!confirm('Delete this webhook?')) return;
    webhooks = webhooks.filter(w => w.id !== id);
    localStorage.setItem('dashboard-webhooks', JSON.stringify(webhooks));
    loadWebhooks();
    showToast('Webhook Deleted', 'Webhook removed successfully', 'success');
}

async function testWebhook(id) {
    const webhook = webhooks.find(w => w.id === id);
    if (!webhook) return;

    const testPayload = {
        event: 'test',
        message: 'Test webhook from PyAI-Slayer Dashboard',
        timestamp: new Date().toISOString()
    };

    const success = await sendWebhook(webhook.url, testPayload);
    if (success) {
        showToast('Webhook Test', 'Test payload sent successfully', 'success');
    }
}

async function sendWebhook(url, payload) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            addNotification('webhook', 'Webhook Sent', `Payload delivered to ${url}`, { payload });
            return true;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('Webhook error:', error);
        showToast('Webhook Failed', `Failed to send to ${url}`, 'error');
        return false;
    }
}

// ===== ACTIVITY TIMELINE =====
let activityLog = JSON.parse(localStorage.getItem('dashboard-activity-log')) || [];

function logActivity(action, details) {
    const activity = {
        id: Date.now(),
        action, // 'test_run', 'filter_applied', 'report_exported', etc.
        details,
        timestamp: new Date().toISOString(),
        user: 'Current User' // In production, get from auth
    };

    activityLog.unshift(activity);

    // Keep only last 200 activities
    if (activityLog.length > 200) {
        activityLog = activityLog.slice(0, 200);
    }

    localStorage.setItem('dashboard-activity-log', JSON.stringify(activityLog));
    updateActivityTimeline();
}

function updateActivityTimeline() {
    const container = document.getElementById('activityTimeline');
    if (!container) return;

    const recentActivities = activityLog.slice(0, 20);

    if (recentActivities.length === 0) {
        container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No recent activity</div>';
        return;
    }

    container.innerHTML = recentActivities.map(activity => `
        <div class="activity-item">
            <div class="activity-icon">${getActivityIcon(activity.action)}</div>
            <div class="activity-content">
                <div class="activity-action">${formatActivityAction(activity.action)}</div>
                <div class="activity-details">${activity.details}</div>
                <div class="activity-time">${formatTimeAgo(activity.timestamp)}</div>
            </div>
        </div>
    `).join('');
}

function getActivityIcon(action) {
    const icons = {
        test_run: '🧪',
        filter_applied: '🔍',
        report_exported: '📄',
        alert_created: '⚠️',
        webhook_added: '🔗',
        theme_changed: '🎨'
    };
    return icons[action] || '📌';
}

function formatActivityAction(action) {
    const map = {
        test_run: 'Test Executed',
        filter_applied: 'Filter Applied',
        report_exported: 'Report Exported',
        alert_created: 'Alert Rule Created',
        webhook_added: 'Webhook Configured',
        theme_changed: 'Theme Changed'
    };
    return map[action] || action;
}

// ===== INITIALIZATION =====
function initCollaborationFeatures() {
    initNotificationCenter();
    loadAlertRules();
    loadWebhooks();
    updateActivityTimeline();

    // Log initial activity
    logActivity('dashboard_loaded', 'Dashboard accessed');
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCollaborationFeatures);
} else {
    initCollaborationFeatures();
}

// Listen for test completions to create notifications
document.addEventListener('test_completed', (event) => {
    const { test, status } = event.detail;
    if (status === 'failed') {
        addNotification('test_failed', 'Test Failed', `${test.name} has failed`, { test });
    }
});
