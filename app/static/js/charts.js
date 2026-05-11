// Chart.js 封装 — 雷达图、K线模拟图、收益曲线等图表工具
window.Charts = {
    createRadar(canvasId, labels, data, label) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: 'rgba(233,69,96,0.2)',
                    borderColor: '#e94560',
                    borderWidth: 2,
                    pointBackgroundColor: '#e94560',
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { stepSize: 20 }
                    }
                }
            }
        });
    },

    createLine(canvasId, labels, datasets) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
            }
        });
    },

    createKLine(canvasId, dates, closes, volumes) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '收盘价',
                    data: closes,
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233,69,96,0.05)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    yAxisID: 'y',
                }]
            },
            options: {
                responsive: true,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    y: {
                        position: 'left',
                        title: { display: true, text: '价格 (元)' }
                    },
                }
            }
        });
    }
};
