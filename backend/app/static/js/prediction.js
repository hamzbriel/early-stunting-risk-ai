/**
 * prediction.js - Prediction form handling and API integration
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const submitBtn = document.getElementById('submit-btn');
    const submitText = document.getElementById('submit-text');
    const submitLoading = document.getElementById('submit-loading');

    form.addEventListener('submit', handleSubmit);

    async function handleSubmit(e) {
        e.preventDefault();
        setLoadingState(true);

        try {
            const formData = new FormData(form);
            const data = {};
            for (const [key, value] of formData.entries()) {
                if (['age_month', 'mother_age', 'mother_working', 'father_working',
                     'clean_water', 'electricity', 'exclusive_breastfeeding',
                     'diarrhea_history'].includes(key)) {
                    data[key] = parseInt(value);
                } else if (['birth_weight', 'birth_length'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Prediction failed');
            }

            const result = await response.json();
            sessionStorage.setItem('prediction_result', JSON.stringify(result));
            window.location.href = '/result';
        } catch (error) {
            alert(`Error: ${error.message}`);
            setLoadingState(false);
        }
    }

    function setLoadingState(loading) {
        if (loading) {
            submitBtn.disabled = true;
            submitText.classList.add('hidden');
            submitLoading.classList.remove('hidden');
            form.classList.add('opacity-75', 'pointer-events-none');
        } else {
            submitBtn.disabled = false;
            submitText.classList.remove('hidden');
            submitLoading.classList.add('hidden');
            form.classList.remove('opacity-75', 'pointer-events-none');
        }
    }
});
