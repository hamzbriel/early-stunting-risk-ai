/**
 * app.js - Global utilities and initialization
 * Early Stunting Risk AI
 */

// Global app namespace
const App = {
    // API base URL (empty string = same origin)
    API_BASE: '',

    // Utility: Show toast notification
    showToast(message, variant = 'info', duration = 5000) {
        const container = document.getElementById('toast-container') || this.createToastContainer();

        const toast = document.createElement('div');
        toast.className = `toast toast-${variant} animate-slide-in`;
        toast.innerHTML = `
            <div class="flex items-start">
                <div class="flex-1">${message}</div>
                <button onclick="this.closest('.toast').remove()" class="ml-3 text-gray-500 hover:text-gray-700">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;

        container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => toast.remove(), duration);
        }

        return toast;
    },

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed top-4 right-4 z-50 space-y-2 max-w-md';
        document.body.appendChild(container);
        return container;
    },

    // Utility: Open modal
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    },

    // Utility: Close modal
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
    },

    // Utility: API fetch wrapper
    async fetch(endpoint, options = {}) {
        const url = `${this.API_BASE}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || error.message || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
};

// Global modal functions (for onclick handlers)
window.openModal = (id) => App.openModal(id);
window.closeModal = (id) => App.closeModal(id);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Early Stunting Risk AI - Initialized');
});
