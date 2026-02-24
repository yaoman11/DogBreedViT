let selectedFile = null;
let currentMode = 'upload'; // 'upload' or 'camera'
let cameraStream = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const cameraInput = document.getElementById('cameraInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const detections = document.getElementById('detections');
const errorMessage = document.getElementById('errorMessage');

// Camera modal elements
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

// Option buttons
const uploadBtn = document.getElementById('uploadBtn');
const cameraBtn = document.getElementById('cameraBtn');

// Mode selection
uploadBtn.addEventListener('click', () => {
    currentMode = 'upload';
    uploadBtn.classList.add('active');
    cameraBtn.classList.remove('active');
    updateUploadAreaUI();
});

cameraBtn.addEventListener('click', () => {
    currentMode = 'camera';
    cameraBtn.classList.add('active');
    uploadBtn.classList.remove('active');
    updateUploadAreaUI();
});

function updateUploadAreaUI() {
    const icon = uploadArea.querySelector('.upload-icon');
    const mainText = uploadArea.querySelector('p:first-of-type');
    const hint = uploadArea.querySelector('.upload-hint');
    
    if (currentMode === 'camera') {
        icon.textContent = '📷';
        mainText.textContent = 'Click to open camera';
        hint.textContent = 'Take a photo directly';
        uploadArea.classList.add('camera-mode');
    } else {
        icon.textContent = '📁';
        mainText.textContent = 'Click to upload or drag and drop';
        hint.textContent = 'Supports: JPG, PNG, JPEG';
        uploadArea.classList.remove('camera-mode');
    }
}

// Check if device is mobile
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

// Check if device is iOS
function isIOSDevice() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

// Click to upload or take photo
uploadArea.addEventListener('click', () => {
    if (currentMode === 'camera') {
        if (isMobileDevice()) {
            // On mobile (including iOS), use native camera input
            cameraInput.click();
        } else {
            // On desktop, use webcam with getUserMedia
            openCamera();
        }
    } else {
        fileInput.click();
    }
});

// Prevent default behavior for touch devices (but allow the click to trigger)
uploadArea.addEventListener('touchstart', (e) => {
    // Don't prevent default on iOS to allow input click
    if (!isIOSDevice()) {
        e.preventDefault();
    }
    if (currentMode === 'camera') {
        if (isMobileDevice()) {
            // iOS needs setTimeout to work properly with hidden inputs
            setTimeout(() => {
                cameraInput.click();
            }, 100);
        } else {
            openCamera();
        }
    } else {
        setTimeout(() => {
            fileInput.click();
        }, 100);
    }
});

// File selection from gallery
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Camera capture
cameraInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Open camera (desktop/laptop)
async function openCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' },
            audio: false 
        });
        
        cameraVideo.srcObject = cameraStream;
        cameraModal.style.display = 'flex';
        
    } catch (error) {
        console.error('Camera error:', error);
        if (error.name === 'NotAllowedError') {
            showError('Camera access denied. Please allow camera permission in your browser settings.');
        } else if (error.name === 'NotFoundError') {
            showError('No camera found on this device.');
        } else {
            showError('Unable to access camera: ' + error.message);
        }
    }
}

// Close camera
function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraVideo.srcObject = null;
    cameraModal.style.display = 'none';
}

// Capture photo from video
captureBtn.addEventListener('click', () => {
    const canvas = cameraCanvas;
    const context = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(cameraVideo, 0, 0);
    
    // Convert canvas to blob
    canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        handleFile(file);
        closeCamera();
    }, 'image/jpeg', 0.95);
});

// Close camera button
closeCameraBtn.addEventListener('click', () => {
    closeCamera();
});

// Drag and drop (only for upload mode)
uploadArea.addEventListener('dragover', (e) => {
    if (currentMode === 'upload') {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }
});

uploadArea.addEventListener('dragleave', () => {
    if (currentMode === 'upload') {
        uploadArea.classList.remove('dragover');
    }
});

uploadArea.addEventListener('drop', (e) => {
    if (currentMode === 'upload') {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    }
});

function handleFile(file) {
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Enable analyze button and show reset button
    analyzeBtn.disabled = false;
    resetBtn.style.display = 'block';
    
    // Hide previous results
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    
    // Reset file inputs for next use
    fileInput.value = '';
    cameraInput.value = '';
}

// Reset button handler
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    analyzeBtn.disabled = true;
    resetBtn.style.display = 'none';
    fileInput.value = '';
    cameraInput.value = '';
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    analyzeBtn.disabled = true;
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loading.style.display = 'none';
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message);
        }
    } catch (error) {
        loading.style.display = 'none';
        showError('Failed to connect to server: ' + error.message);
    }
    
    analyzeBtn.disabled = false;
});

function displayResults(data) {
    // Show result image
    resultImage.src = data.result_image + '?t=' + new Date().getTime();
    resultsSection.style.display = 'block';
    
    // Scroll to results on mobile
    if (window.innerWidth <= 768) {
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
    
    // Clear previous detections
    detections.innerHTML = '';
    
    // Display each detection
    data.detections.forEach((det) => {
        const card = document.createElement('div');
        card.className = 'detection-card' + (!det.is_valid ? ' unknown' : '');
        
        // Add note for unknown breed
        const unknownNote = !det.is_valid ? 
            '<p class="unknown-note">⚠️ Breed not recognized. This may be a dog breed not in our trained classes.</p>' : '';
        
        const confidenceBadge = det.is_valid ? 
            `<div class="confidence-badge">${det.confidence.toFixed(1)}%</div>` : '';
        
        card.innerHTML = `
            <div class="detection-header">
                <div class="detection-title">
                    Dog #${det.id}: ${det.breed}
                </div>
                ${confidenceBadge}
            </div>
            
            ${unknownNote}
        `;
        
        detections.appendChild(card);
    });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}
