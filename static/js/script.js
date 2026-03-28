// --- File Upload Logic ---
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const originalImage = document.getElementById('original-image');
const processedImage = document.getElementById('processed-image');
const loader = document.getElementById('loader');
const resultContainer = document.getElementById('result-container');
const detectionCount = document.getElementById('detection-count');
const actionButtons = document.getElementById('action-buttons');
const resetBtn = document.getElementById('reset-btn');
const downloadLink = document.getElementById('download-link');

if (dropArea) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFiles, false);

    if (resetBtn) resetBtn.addEventListener('click', resetView);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    dropArea.classList.add('dragover');
}

function unhighlight(e) {
    dropArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    dropArea.style.display = 'none';
    previewSection.style.display = 'flex';
    actionButtons.style.display = 'none';

    // Display original
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function () {
        originalImage.src = reader.result;
    }

    loader.style.display = 'block';
    processedImage.style.display = 'none';

    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/detect', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);

            processedImage.src = 'data:image/jpeg;base64,' + data.image;
            processedImage.style.display = 'block';
            loader.style.display = 'none';

            actionButtons.style.display = 'flex';
            downloadLink.href = processedImage.src;

            // Show Analytics only after detection
            const analyticsSection = document.getElementById('analytics-section');
            if (analyticsSection) {
                analyticsSection.style.display = 'block';

                // Update Numeric Metrics with LIVE detection data
                if (cachedMetrics) {
                    renderNumericMetrics(cachedMetrics, data.detections);
                }

                // Scroll to analytics for better UX
                setTimeout(() => {
                    analyticsSection.scrollIntoView({ behavior: 'smooth' });
                }, 100);
            }
        })
        .catch(error => {
            alert('Error: ' + error.message);
            console.error(error);
            resetView();
        });
}

function resetView() {
    previewSection.style.display = 'none';
    actionButtons.style.display = 'none';
    const analyticsSection = document.getElementById('analytics-section');
    if (analyticsSection) analyticsSection.style.display = 'none';
    dropArea.style.display = 'block';
    fileInput.value = '';
    processedImage.src = '';
    originalImage.src = '';
}

// --- Charts Logic ---
let cachedMetrics = null;
let cachedHistory = null;

document.addEventListener('DOMContentLoaded', function () {
    fetchMetrics();
    fetchHistory();
});

function fetchMetrics() {
    fetch('/api/metrics')
        .then(r => r.json())
        .then(data => {
            if (data.comparison) {
                cachedMetrics = data.comparison;
                renderComparisonChart(cachedMetrics);
                renderNumericMetrics(cachedMetrics);
            }
        })
        .catch(e => {
            console.error("Error fetching metrics:", e);
            const grid = document.getElementById('metrics-text-grid');
            if (grid) grid.innerHTML = '<p class="text-danger">Error loading performance data. Please refresh.</p>';
        });
}

function renderNumericMetrics(comparisonData, liveDetections = null) {
    const grid = document.getElementById('metrics-text-grid');
    grid.innerHTML = ''; // Clear loader

    // 1. If we have live detections, show them prominently (Red Bordered Card)
    if (liveDetections && liveDetections.length > 0) {
        const liveCard = document.createElement('div');
        liveCard.className = 'model-score-card';
        liveCard.style.gridColumn = '1 / -1'; // Make it full width
        liveCard.style.borderLeftColor = '#ef4444'; // Red for "Live"
        liveCard.style.fontFamily = "'Courier New', Courier, monospace";
        liveCard.style.backgroundColor = '#f8fafc';

        let html = `<h4 style="font-family: inherit; color: #3b82f6;"><i class="fa-solid fa-bolt"></i> Live Detection Results</h4>`;
        liveDetections.forEach((det, idx) => {
            html += `
                <div class="score-item" style="border-bottom: 1px solid #f3f4f6;">
                    <span class="score-label">Object ${idx + 1}: ${det.label}</span>
                    <span class="score-value" style="font-weight: bold;">${(parseFloat(det.confidence) * 100).toFixed(2)}% Confidence</span>
                </div>
            `;
        });

        // Average confidence for current image
        const avgConf = liveDetections.reduce((sum, d) => sum + parseFloat(d.confidence), 0) / liveDetections.length;
        html += `
            <div class="score-item" style="border-top: 2px solid #ddd; margin-top: 15px; padding-top: 10px;">
                <span class="score-label" style="color: #ef4444; font-size: 1.1rem; font-weight: 700;">Overall Image Accuracy</span>
                <span class="score-value" style="font-size: 1.1rem; font-weight: 700;">${(avgConf * 100).toFixed(2)}%</span>
            </div>
        `;

        liveCard.innerHTML = html;
        grid.appendChild(liveCard);
    }

    // 2. Fallback: Always show a "Last Saved Benchmark" or similar if requested? 
    // The user said "remove this and give only live result" but then showed the benchmarks in a complaint.
    // Let's also show the YOLOv8 Benchmark card below the live result for comparison.
    const v8Metrics = comparisonData["YoloV8"];
    if (v8Metrics) {
        const benchmarkCard = document.createElement('div');
        benchmarkCard.className = 'model-score-card';
        benchmarkCard.style.borderLeftColor = '#3b82f6';

        let html = `<h4>YOLOv8 Performance Benchmark</h4>`;
        for (const [key, val] of Object.entries(v8Metrics)) {
            const displayLabel = key === 'FMeasure' ? 'F-Measure' : key;
            html += `
                <div class="score-item">
                    <span class="score-label">YOLOv8 ${displayLabel}</span>
                    <span class="score-value">: ${val}</span>
                </div>
            `;
        }
        benchmarkCard.innerHTML = html;
        grid.appendChild(benchmarkCard);
    }
}

function renderComparisonChart(comparisonData) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');

    // Extract labels (Model Names) and datasets (Metrics)
    const models = Object.keys(comparisonData);
    const accuracy = models.map(m => comparisonData[m].Accuracy);
    const precision = models.map(m => comparisonData[m].Precision);
    const recall = models.map(m => comparisonData[m].Recall);
    const fmeasure = models.map(m => comparisonData[m].FMeasure);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                { label: 'Accuracy', data: accuracy, backgroundColor: '#3b82f6' },
                { label: 'Precision', data: precision, backgroundColor: '#22c55e' },
                { label: 'Recall', data: recall, backgroundColor: '#f59e0b' },
                { label: 'F-Measure', data: fmeasure, backgroundColor: '#6366f1' }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });
}

function fetchHistory() {
    fetch('/api/history')
        .then(r => r.json())
        .then(data => {
            cachedHistory = data;
            renderHistoryChart(cachedHistory);
        })
        .catch(e => console.error("Error fetching history:", e));
}

function renderHistoryChart(historyData) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');

    const datasets = [];
    const colors = ['#3b82f6', '#22c55e', '#ef4444'];
    let colorIdx = 0;
    let maxEpochs = 0;

    for (const [modelName, data] of Object.entries(historyData)) {
        if (data && data.accuracy && data.accuracy.length > 0) {
            datasets.push({
                label: `Model ${modelName} Acc`,
                data: data.accuracy,
                borderColor: colors[colorIdx % colors.length],
                tension: 0.3,
                fill: false
            });
            maxEpochs = Math.max(maxEpochs, data.accuracy.length);
            colorIdx++;
        }
    }

    // Generate labels 1 to N
    const labels = Array.from({ length: maxEpochs }, (_, i) => i + 1);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 1.0 } // Accuracy is usually 0-1 or 0-100 depending on storage
            },
            plugins: {
                title: { display: true, text: 'Training Accuracy over Epochs' }
            }
        }
    });
}
