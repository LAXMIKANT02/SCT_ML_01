const ctx = document.getElementById('priceChart').getContext('2d');
const priceChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
        datasets: [{
            label: 'Predicted Price',
            data: [120000, 150000, 170000, 130000, 160000, 180000, 200000],
            backgroundColor: 'rgba(255, 215, 0, 0.2)',
            borderColor: 'rgba(255, 215, 0, 1)',
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
