AOS.init({
    duration: 800,
    once: true
});


window.addEventListener('load', () => {
    const form = document.querySelector('.form-container');
    if (form) form.style.transform = 'scale(1)';
});
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('themeToggle');
    const icon = toggleBtn.querySelector('i');

    toggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
        icon.classList.toggle('fa-moon');
        icon.classList.toggle('fa-sun');
    });
});
