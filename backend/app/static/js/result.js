/**
 * result.js - Populates prediction result page from sessionStorage
 */

document.addEventListener('DOMContentLoaded', () => {
    const resultData = sessionStorage.getItem('prediction_result');

    if (!resultData) {
        console.log('No prediction result found.');
        return;
    }

    try {
        const result = JSON.parse(resultData);

        // Determine risk level and colors
        const riskConfig = getRiskConfig(result.prediction);

        // Update prediction with color
        const predictionEl = document.getElementById('res-prediction');
        predictionEl.textContent = result.prediction;
        predictionEl.className = `text-6xl font-extrabold mb-3 ${riskConfig.textColor}`;

        // Update confidence
        const confidenceEl = document.getElementById('res-confidence');
        confidenceEl.textContent = `${result.confidence.toFixed(1)}%`;
        confidenceEl.className = `text-2xl font-bold ${riskConfig.textColor}`;

        // Update prediction card border
        const cardEl = document.getElementById('prediction-card');
        cardEl.className = `bg-white p-8 rounded-2xl shadow-xl mb-6 text-center border-t-4 ${riskConfig.borderColor}`;

        // Populate probabilities with colors
        const probContainer = document.getElementById('res-probabilities');
        probContainer.innerHTML = '';
        for (const [cls, prob] of Object.entries(result.probabilities)) {
            const clsConfig = getRiskConfig(cls);
            probContainer.innerHTML += `
                <div class="mb-3">
                    <div class="flex justify-between text-sm font-medium mb-1">
                        <span class="${clsConfig.textColor}">${cls}</span>
                        <span class="${clsConfig.textColor}">${prob.toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 h-3 rounded-full overflow-hidden">
                        <div class="${clsConfig.bgColor} h-3 rounded-full transition-all duration-500" style="width: ${prob}%"></div>
                    </div>
                </div>
            `;
        }

        // Populate recommendations as list
        const recContainer = document.getElementById('res-recommendation');
        const recommendations = parseRecommendations(result.recommendation);
        recContainer.innerHTML = '';
        recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.className = 'flex items-start gap-2 text-sm';
            li.innerHTML = `
                <svg class="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span>${rec}</span>
            `;
            recContainer.appendChild(li);
        });

        document.getElementById('loading-message').classList.add('hidden');
        document.getElementById('result-content').classList.remove('hidden');
    } catch (e) {
        console.error('Failed to parse result:', e);
    }
});

/**
 * Get color configuration based on risk level
 */
function getRiskConfig(riskLevel) {
    const level = riskLevel.toLowerCase();

    if (level.includes('low') || level.includes('rendah')) {
        return {
            textColor: 'text-emerald-600',
            bgColor: 'bg-emerald-600',
            borderColor: 'border-emerald-500'
        };
    } else if (level.includes('medium') || level.includes('sedang')) {
        return {
            textColor: 'text-amber-600',
            bgColor: 'bg-amber-600',
            borderColor: 'border-amber-500'
        };
    } else if (level.includes('high') || level.includes('tinggi')) {
        return {
            textColor: 'text-red-700',
            bgColor: 'bg-red-700',
            borderColor: 'border-red-600'
        };
    }

    // Default
    return {
        textColor: 'text-gray-700',
        bgColor: 'bg-gray-600',
        borderColor: 'border-gray-500'
    };
}

/**
 * Parse recommendation text into list items
 */
function parseRecommendations(text) {
    if (!text) return [];

    // Split by common delimiters: periods followed by capital letters, newlines, or numbered lists
    let items = text
        .split(/(?<=[.!])\s+(?=[A-Z])|[\n\r]+|\d+\.\s+/)
        .map(item => item.trim())
        .filter(item => item.length > 0);

    // If splitting resulted in too few items (text might be a single sentence), just return as one item
    if (items.length === 1) {
        return items;
    }

    // Remove trailing periods for cleaner list display
    items = items.map(item => item.replace(/\.$/, ''));

    return items;
}
