/**
 * validation.js - Client-side form validation
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    // Add real-time validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('blur', validateInput);
        input.addEventListener('input', clearError);
    });

    function validateInput(e) {
        const input = e.target;
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);

        let errorMessage = '';

        if (input.value === '') {
            errorMessage = 'Field ini wajib diisi';
        } else if (isNaN(value)) {
            errorMessage = 'Nilai harus berupa angka';
        } else if (min !== undefined && value < min) {
            errorMessage = `Nilai minimal: ${min}`;
        } else if (max !== undefined && value > max) {
            errorMessage = `Nilai maksimal: ${max}`;
        }

        if (errorMessage) {
            showError(input, errorMessage);
        } else {
            clearError({ target: input });
        }
    }

    function showError(input, message) {
        clearError({ target: input });

        input.classList.add('border-red-500', 'focus:ring-red-500');

        const errorDiv = document.createElement('p');
        errorDiv.className = 'form-error';
        errorDiv.textContent = message;

        input.parentElement.appendChild(errorDiv);
    }

    function clearError(e) {
        const input = e.target;
        input.classList.remove('border-red-500', 'focus:ring-red-500');

        const errorDiv = input.parentElement.querySelector('.form-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
});
