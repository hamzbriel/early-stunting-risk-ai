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

        // Update DOM elements
        document.getElementById('res-prediction').textContent = result.prediction;
        document.getElementById('res-confidence').textContent = `${result.confidence.toFixed(1)}%`;

        // Populate probabilities
        const probContainer = document.getElementById('res-probabilities');
        probContainer.innerHTML = '';
        for (const [cls, prob] of Object.entries(result.probabilities)) {
            probContainer.innerHTML += `
                <div class="mb-2">
                    <div class="flex justify-between text-sm">
                        <span>${cls}</span><span>${prob.toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 h-2 rounded"><div class="bg-emerald-600 h-2 rounded" style="width: ${prob}%"></div></div>
                </div>
            `;
        }

        // Populate recommendations
        document.getElementById('res-recommendation').textContent = result.recommendation;

        document.getElementById('loading-message').classList.add('hidden');
        document.getElementById('result-content').classList.remove('hidden');
    } catch (e) {
        console.error('Failed to parse result:', e);
    }
});
