/**
 * main.js - Trading Bot Web Arayüzü
 * Genel yardımcı fonksiyonlar ve bot kontrol işlemleri
 */

'use strict';

// ============================================================
// API Yardımcıları
// ============================================================

/**
 * GET isteği gönderir ve JSON döndürür.
 */
async function apiGet(url) {
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error(`API GET hatası (${url}):`, e);
        return { success: false, error: e.message };
    }
}

/**
 * POST isteği gönderir ve JSON döndürür.
 */
async function apiPost(url, data = {}) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error(`API POST hatası (${url}):`, e);
        return { success: false, error: e.message };
    }
}

// ============================================================
// Alert / Bildirim
// ============================================================

/**
 * Ekranda bildirim gösterir.
 * type: 'success' | 'danger' | 'warning' | 'info'
 */
function showAlert(type, message, duration = 4000) {
    const container = document.getElementById('alertContainer');
    if (!container) return;

    const alertId = 'alert-' + Date.now();
    const icons = {
        success: 'bi-check-circle',
        danger: 'bi-exclamation-triangle',
        warning: 'bi-exclamation-circle',
        info: 'bi-info-circle',
    };
    const icon = icons[type] || 'bi-info-circle';

    const alertEl = document.createElement('div');
    alertEl.id = alertId;
    alertEl.className = `alert alert-${type} alert-dismissible d-flex align-items-center mb-2`;
    alertEl.innerHTML = `
        <i class="bi ${icon} me-2"></i>
        <span>${message}</span>
        <button type="button" class="btn-close btn-close-white ms-auto" onclick="document.getElementById('${alertId}').remove()"></button>
    `;

    container.appendChild(alertEl);

    // Otomatik kaldır
    if (duration > 0) {
        setTimeout(() => {
            const el = document.getElementById(alertId);
            if (el) {
                el.style.opacity = '0';
                el.style.transition = 'opacity 0.3s';
                setTimeout(() => el.remove(), 300);
            }
        }, duration);
    }
}

// ============================================================
// Sayı Formatlama
// ============================================================

/**
 * Sayıyı para birimi formatına çevirir.
 */
function formatCurrency(value, decimals = 4) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    if (isNaN(num)) return '--';
    const prefix = num >= 0 ? '' : '';
    return `${prefix}${num.toFixed(decimals)} USDT`;
}

/**
 * Yüzde formatı.
 */
function formatPct(value, decimals = 2) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    if (isNaN(num)) return '--';
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
}

/**
 * Büyük sayıları kısaltır (1000 -> 1K).
 */
function formatLarge(value) {
    if (value >= 1e6) return (value / 1e6).toFixed(2) + 'M';
    if (value >= 1e3) return (value / 1e3).toFixed(2) + 'K';
    return value.toFixed(2);
}

// ============================================================
// Bot Kontrol
// ============================================================

let _botActionInProgress = false;

/**
 * Bot aksiyonu gönderir: start, stop, close-all
 */
async function botAction(action) {
    if (_botActionInProgress) return;

    const confirmMessages = {
        'close-all': 'Tüm açık pozisyonları kapatmak istediğinizden emin misiniz?',
        'stop': 'Botu durdurmak istediğinizden emin misiniz?',
    };

    if (confirmMessages[action] && !confirm(confirmMessages[action])) return;

    _botActionInProgress = true;

    // Butonları devre dışı bırak
    const btns = ['btnStart', 'btnStop', 'btnCloseAll'];
    btns.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = true;
    });

    try {
        const result = await apiPost(`/api/bot/${action}`);

        if (result.success) {
            showAlert('success', result.message || 'İşlem tamamlandı.');
        } else {
            showAlert('danger', result.error || result.message || 'Hata oluştu.');
        }
    } catch (e) {
        showAlert('danger', `Bağlantı hatası: ${e.message}`);
    } finally {
        _botActionInProgress = false;
        btns.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = false;
        });
    }
}

// ============================================================
// Sidebar Toggle
// ============================================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('mainContent');

    if (window.innerWidth <= 768) {
        // Mobil: sidebar göster/gizle
        sidebar.classList.toggle('open');
    } else {
        // Masaüstü: sidebar genişliğini değiştir
        if (sidebar.style.transform === 'translateX(-100%)') {
            sidebar.style.transform = '';
            main.style.marginLeft = '';
        } else {
            sidebar.style.transform = 'translateX(-100%)';
            main.style.marginLeft = '0';
        }
    }
}

// Mobil'de sidebar dışına tıklayınca kapat
document.addEventListener('click', function(e) {
    if (window.innerWidth > 768) return;
    const sidebar = document.getElementById('sidebar');
    const toggle = document.querySelector('.sidebar-toggle');
    if (sidebar && !sidebar.contains(e.target) && toggle && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
    }
});

// ============================================================
// Klavye Kısayolları
// ============================================================

document.addEventListener('keydown', function(e) {
    // Ctrl+Alt+S = Durdur
    if (e.ctrlKey && e.altKey && e.key === 's') {
        botAction('stop');
    }
    // Ctrl+Alt+R = Başlat
    if (e.ctrlKey && e.altKey && e.key === 'r') {
        botAction('start');
    }
});

// ============================================================
// Renk Yardımcıları
// ============================================================

/**
 * Pozitif değer için yeşil, negatif için kırmızı CSS sınıfı.
 */
function pnlClass(value) {
    return value >= 0 ? 'text-success' : 'text-danger';
}

/**
 * PNL HTML badge'i.
 */
function pnlBadge(value, suffix = ' USDT') {
    const cls = value >= 0 ? 'text-success' : 'text-danger';
    const sign = value >= 0 ? '+' : '';
    return `<span class="${cls} fw-bold">${sign}${value.toFixed(4)}${suffix}</span>`;
}

// ============================================================
// Zaman Yardımcıları
// ============================================================

function timeAgo(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);

    if (diffSec < 60) return `${diffSec}sn önce`;
    if (diffMin < 60) return `${diffMin}dk önce`;
    if (diffHour < 24) return `${diffHour}sa önce`;
    return date.toLocaleDateString('tr-TR');
}

function formatDuration(startStr) {
    if (!startStr) return '--';
    const start = new Date(startStr);
    const now = new Date();
    const diffSec = Math.floor((now - start) / 1000);
    const h = Math.floor(diffSec / 3600);
    const m = Math.floor((diffSec % 3600) / 60);
    const s = diffSec % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ============================================================
// Toast Bildirimleri (Bootstrap 5)
// ============================================================

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) return;

    const toastId = 'toast-' + Date.now();
    const bgClasses = {
        success: 'bg-success',
        danger: 'bg-danger',
        warning: 'bg-warning text-dark',
        info: 'bg-info',
    };

    const toastEl = document.createElement('div');
    toastEl.id = toastId;
    toastEl.className = `toast ${bgClasses[type] || ''} text-white border-0`;
    toastEl.setAttribute('role', 'alert');
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toastEl);

    const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 3000 });
    toast.show();

    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

// ============================================================
// Tablo Sıralama (Basit)
// ============================================================

function sortTable(tableId, columnIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    const isAsc = table.getAttribute('data-sort-dir') !== 'asc';
    table.setAttribute('data-sort-dir', isAsc ? 'asc' : 'desc');

    rows.sort((a, b) => {
        const aVal = a.cells[columnIndex]?.textContent.trim() || '';
        const bVal = b.cells[columnIndex]?.textContent.trim() || '';

        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return isAsc ? aNum - bNum : bNum - aNum;
        }
        return isAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });

    rows.forEach(row => tbody.appendChild(row));
}

// ============================================================
// Başlangıç
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    // Tooltip'leri aktive et
    const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipEls.forEach(el => new bootstrap.Tooltip(el));

    // Toast container
    if (!document.getElementById('toastContainer')) {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }

    console.log('Trading Bot UI yüklendi.');
});
