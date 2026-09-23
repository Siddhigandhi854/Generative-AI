// GAN Face Generator Dashboard Client Logic (Connected to FastAPI Backend API)

const API_BASE_URL = (typeof window !== 'undefined' && window.API_BASE_URL && window.API_BASE_URL !== window.location.origin) ? window.API_BASE_URL : '';

let activeSeed = Math.floor(Math.random() * 900000) + 1000;
let activeCount = 1;
let currentPreviewUrl = null;
let isCrispPixels = true;

// Session Gallery State
let gallerySessionState = [];

document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    updateSeedDisplay(activeSeed);
    renderGallery();
    loadModelInfo();
    
    // Auto-generate initial synthetic face on startup for instant preview
    generateFaces(1);

    // Keyboard Event Listener for Modal Close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });
});

// FETCH WITH TIMEOUT HELPER (AbortController)
async function fetchWithTimeout(resource, options = {}, timeoutMs = 15000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (err) {
        clearTimeout(id);
        if (err.name === 'AbortError') {
            throw new Error(`Connection timed out after ${timeoutMs / 1000}s. The server may be processing a heavy load.`);
        }
        throw new Error("Unable to connect to the backend server. Please verify the server is running.");
    }
}

// 1. BACKEND HEALTH CHECK
async function checkBackendHealth() {
    const deviceTag = document.getElementById('backend-device-tag');
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 5000);
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                const devStr = data.device.toUpperCase();
                updateStatusBadge(`Status: Ready (${devStr})`, 'emerald');
                if (deviceTag) {
                    deviceTag.innerHTML = `
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                        ONLINE (${devStr})
                    `;
                    deviceTag.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm flex items-center gap-1";
                }
            } else {
                updateStatusBadge("Status: Model Unloaded", "rose");
                if (deviceTag) {
                    deviceTag.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-rose-400"></span> UNLOADED`;
                    deviceTag.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1";
                }
                showError("The PyTorch generator model file 'generator.pth' is missing or failed to load on the server.");
            }
        }
    } catch (e) {
        console.warn("Backend API server not reachable:", e.message);
        updateStatusBadge("Status: Server Offline", "rose");
        if (deviceTag) {
            deviceTag.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span> OFFLINE`;
            deviceTag.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1";
        }
    }
}

// 2. DYNAMIC MODEL SPECIFICATIONS
async function loadModelInfo() {
    const grid = document.getElementById('model-info-grid');
    if (!grid) return;

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/model-info`, {}, 8000);
        if (!response.ok) throw new Error("Failed to fetch model information.");

        const data = await response.json();

        grid.innerHTML = `
            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-indigo-400 uppercase tracking-wider block">Model Type</span>
                <h3 class="text-lg font-bold text-white">${data.model_type || 'DCGAN'}</h3>
                <p class="text-xs text-slate-400">Deep Convolutional Generative Adversarial Network</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-purple-400 uppercase tracking-wider block">Framework</span>
                <h3 class="text-lg font-bold text-white">${data.framework || 'PyTorch'}</h3>
                <p class="text-xs text-slate-400">Deep Learning Backend Framework</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider block">Generator Architecture</span>
                <h3 class="text-sm font-bold text-white leading-snug">${data.generator_architecture || '5 Transposed Convolutions'}</h3>
                <p class="text-xs text-slate-400">Layer Stack Architecture</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider block">Image Resolution & Channels</span>
                <h3 class="text-lg font-bold text-white">${data.image_dimensions?.resolution || '64 x 64 pixels'}</h3>
                <p class="text-xs text-slate-400">${data.image_dimensions?.channels || 3} Channels (${data.image_dimensions?.format || 'RGB'})</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-amber-400 uppercase tracking-wider block">Latent Dimension</span>
                <h3 class="text-lg font-bold text-white">${data.latent_dimension || 100} Dimensions</h3>
                <p class="text-xs text-slate-400">Gaussian Noise Input Vector Z ~ N(0, 1)</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-blue-400 uppercase tracking-wider block">Saved Model File</span>
                <h3 class="text-lg font-bold font-mono text-white">${data.saved_model_file || 'generator.pth'}</h3>
                <p class="text-xs text-slate-400">State Dictionary Checkpoint</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md">
                <span class="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider block">Inference Status</span>
                <h3 class="text-lg font-bold text-emerald-400">${data.inference_status || 'Active'}</h3>
                <p class="text-xs text-slate-400">Backend Server Inference Engine</p>
            </div>

            <div class="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 space-y-2.5 shadow-xl backdrop-blur-md opacity-80">
                <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Training Dataset</span>
                <h3 class="text-lg font-bold text-slate-300">${data.training_dataset || 'Not available'}</h3>
                <p class="text-xs text-slate-500">Not recorded in model state dict</p>
            </div>
        `;
    } catch (err) {
        console.error("Failed to load model specifications:", err);
        grid.innerHTML = `<div class="col-span-full text-center text-rose-400 p-6 bg-slate-900/60 border border-slate-800 rounded-xl">Unable to fetch model specifications from backend server.</div>`;
    }
}

// 3. TAB SWITCHING WITH ACCESSIBILITY
function switchTab(tabId) {
    const tabs = ['generate', 'about', 'model'];
    tabs.forEach(tab => {
        const section = document.getElementById(`section-${tab}`);
        const button = document.getElementById(`tab-${tab}`);
        if (!section || !button) return;

        if (tab === tabId) {
            section.classList.remove('hidden');
            button.setAttribute('aria-selected', 'true');
            button.className = "px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-indigo-600 text-white shadow-md shadow-indigo-600/30 focus:outline-none focus:ring-2 focus:ring-indigo-400";
        } else {
            section.classList.add('hidden');
            button.setAttribute('aria-selected', 'false');
            button.className = "px-4 py-2 rounded-lg font-medium transition-all duration-200 text-slate-400 hover:text-white hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-indigo-400";
        }
    });

    if (tabId === 'model') {
        loadModelInfo();
    }
}

// 4. RANDOMIZE SEED FUNCTION
function randomizeVector() {
    activeSeed = Math.floor(Math.random() * 900000) + 1000;
    updateSeedDisplay(activeSeed);
    generateFaces(activeCount);
}

function updateSeedDisplay(seed) {
    const seedEl = document.getElementById('active-seed-display');
    if (seedEl) seedEl.textContent = `#${seed}`;
}

// 5. GENERATE FACES API CALL WITH ERROR HANDLING & TIMEOUT
async function generateFaces(count = 1, append = false) {
    activeCount = count;

    hidePlaceholder();
    hideImagePreview();
    hideError();
    showLoading();
    renderGallerySkeletons(count);

    try {
        if (count === 1) {
            const response = await fetchWithTimeout(`${API_BASE_URL}/generate?seed=${activeSeed}&response_format=json`, {
                method: 'POST'
            }, 15000);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ message: `Server error (${response.status})` }));
                throw new Error(errData.message || errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            if (!data.success || !data.image) throw new Error("Invalid response format received from backend API.");

            currentPreviewUrl = data.image;

            const newItem = {
                id: 'face_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
                seed: activeSeed,
                imageUri: data.image,
                timestamp: new Date().toLocaleTimeString()
            };

            if (append) {
                gallerySessionState.push(newItem);
            } else {
                gallerySessionState = [newItem];
            }

            renderSinglePreview(currentPreviewUrl, activeSeed);
            updateStatusBadge("Status: Generated (1 Face)", "emerald");

        } else {
            const response = await fetchWithTimeout(`${API_BASE_URL}/generate-batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: count, seed: activeSeed })
            }, 25000);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ message: `Batch server error (${response.status})` }));
                throw new Error(errData.message || errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            if (!data.success || !data.images) throw new Error("Invalid batch response received from backend API.");

            const images = data.images;
            currentPreviewUrl = images[0];

            const newItems = images.map((imgUri, idx) => ({
                id: 'face_' + Date.now() + '_' + idx + '_' + Math.random().toString(36).substr(2, 4),
                seed: activeSeed + idx,
                imageUri: imgUri,
                timestamp: new Date().toLocaleTimeString()
            }));

            if (append) {
                gallerySessionState = [...gallerySessionState, ...newItems];
            } else {
                gallerySessionState = newItems;
            }

            renderBatchGridPreview(images, count, activeSeed);
            updateStatusBadge(`Status: Generated (${count} Faces)`, "emerald");
        }

        hideLoading();
        showImagePreview();
        renderGallery();

    } catch (err) {
        console.error("API Generation Error:", err);
        hideLoading();
        renderGallery();
        showError(err.message || "Failed to generate face image. Please check backend server.");
    }
}

// 6. GENERATE MORE FACES
function generateMoreFaces(count = 4) {
    activeSeed = Math.floor(Math.random() * 900000) + 1000;
    updateSeedDisplay(activeSeed);
    generateFaces(count, true);
}

// 7. REGENERATE SINGLE GALLERY ITEM
async function regenerateSingleItem(itemId) {
    const itemIdx = gallerySessionState.findIndex(item => item.id === itemId);
    if (itemIdx === -1) return;

    const newSeed = Math.floor(Math.random() * 900000) + 1000;
    const cardEl = document.getElementById(`gallery-item-${itemId}`);
    
    if (cardEl) {
        cardEl.innerHTML = `
            <div class="aspect-square bg-slate-950 rounded-lg flex flex-col items-center justify-center p-4 text-center space-y-2 animate-pulse border border-indigo-500/30">
                <div class="w-8 h-8 rounded-full border-2 border-indigo-500/20 border-t-indigo-500 animate-spin"></div>
                <span class="text-[10px] text-indigo-400 font-mono">Sampling Seed #${newSeed}...</span>
            </div>
        `;
    }

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/generate?seed=${newSeed}&response_format=json`, {
            method: 'POST'
        }, 15000);

        if (!response.ok) throw new Error("Regeneration failed");

        const data = await response.json();
        gallerySessionState[itemIdx].imageUri = data.image;
        gallerySessionState[itemIdx].seed = newSeed;

        renderGallery();
    } catch (err) {
        console.error("Regeneration error:", err.message);
        renderGallery();
        showError(err.message || "Regeneration failed.");
    }
}

// 8. CLEAR GALLERY
function clearGallery() {
    gallerySessionState = [];
    renderGallery();
}

// 9. DOWNLOAD SINGLE IMAGE (gan_face_001.png)
function downloadSingleFace(imgUri, index = 1) {
    const filename = `gan_face_${String(index).padStart(3, '0')}.png`;
    triggerBlobDownload(imgUri, filename);
}

// 10. DOWNLOAD PREVIEW IMAGE
function downloadPreviewImage() {
    if (!currentPreviewUrl) {
        generateFaces(1).then(() => {
            if (currentPreviewUrl) {
                downloadSingleFace(currentPreviewUrl, 1);
            }
        });
        return;
    }
    downloadSingleFace(currentPreviewUrl, 1);
}

// 11. DOWNLOAD ALL GALLERY IMAGES AS ZIP
async function downloadAllGalleryImages() {
    if (gallerySessionState.length === 0) {
        await generateFaces(4);
    }
    if (gallerySessionState.length === 0) return;

    updateStatusBadge("Status: Creating ZIP Archive...", "emerald");

    try {
        const payload = {
            images: gallerySessionState.map(item => item.imageUri)
        };

        const response = await fetchWithTimeout(`${API_BASE_URL}/download-zip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }, 30000);

        if (!response.ok) {
            throw new Error("Failed to create ZIP download on server.");
        }

        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = 'gan_faces_batch.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);

        updateStatusBadge(`Status: Downloaded ${gallerySessionState.length} Images (ZIP)`, "emerald");
    } catch (err) {
        console.error("ZIP download error:", err.message);
        showError(err.message || "Failed to download ZIP archive.");
    }
}

// Helper to trigger browser download
function triggerBlobDownload(dataUri, filename) {
    try {
        let blob;
        if (dataUri.startsWith('data:image')) {
            const parts = dataUri.split(',');
            const mime = parts[0].match(/:(.*?);/)[1];
            const bstr = atob(parts[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            blob = new Blob([u8arr], { type: mime });
        } else {
            const link = document.createElement('a');
            link.href = dataUri;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            return;
        }

        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);
    } catch (e) {
        console.error("Download error:", e);
        showError("Failed to trigger image download.");
    }
}

// 12. PIXEL SCALING MODE TOGGLE (Crisp vs Smooth)
function togglePixelMode() {
    isCrispPixels = !isCrispPixels;
    const mainImg = document.getElementById('main-preview-img');
    const modalImg = document.getElementById('modal-img');
    const label = document.getElementById('pixel-toggle-label');

    if (mainImg) {
        mainImg.className = isCrispPixels ? 'w-full h-full object-cover crisp-pixel-rendering' : 'w-full h-full object-cover smooth-pixel-rendering';
    }
    if (modalImg) {
        modalImg.className = isCrispPixels ? 'w-full h-full object-cover crisp-pixel-rendering' : 'w-full h-full object-cover smooth-pixel-rendering';
    }
    if (label) {
        label.textContent = isCrispPixels ? 'Crisp Pixels' : 'Smooth Pixels';
    }
}

// 13. LIGHTBOX MODAL CONTROLS
function openPreviewLightbox() {
    if (!currentPreviewUrl) return;
    openItemLightbox(currentPreviewUrl, activeSeed);
}

function openItemLightbox(imgUri, seed = 42) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const seedBadge = document.getElementById('modal-seed-badge');
    const downloadBtn = document.getElementById('modal-download-btn');

    if (!modal || !modalImg) return;

    modalImg.src = imgUri;
    if (seedBadge) seedBadge.textContent = `Seed #${seed}`;
    if (downloadBtn) {
        downloadBtn.onclick = () => downloadSingleFace(imgUri, 1);
    }

    modal.classList.remove('hidden');
}

function closeLightbox() {
    const modal = document.getElementById('image-modal');
    if (modal) modal.classList.add('hidden');
}

// UI RENDERING HELPERS

function renderSinglePreview(base64Uri, seed) {
    const container = document.getElementById('preview-image-container');
    if (!container) return;
    const renderingClass = isCrispPixels ? 'crisp-pixel-rendering' : 'smooth-pixel-rendering';
    container.innerHTML = `
        <img id="main-preview-img" src="${base64Uri}" alt="Generated Synthetic Face" class="w-full h-full object-cover ${renderingClass}">
    `;
    const captionEl = document.getElementById('preview-image-caption');
    if (captionEl) captionEl.textContent = `Latent Vector Seed #${seed} • PyTorch DCGAN (64×64 RGB)`;
}

function renderBatchGridPreview(images, count, seed) {
    const container = document.getElementById('preview-image-container');
    if (!container) return;
    const colsClass = count <= 4 ? 'grid-cols-2' : 'grid-cols-3';
    const renderingClass = isCrispPixels ? 'crisp-pixel-rendering' : 'smooth-pixel-rendering';
    
    let gridHtml = `<div class="grid ${colsClass} gap-1.5 w-full h-full p-2 bg-slate-950">`;
    images.forEach((imgUri, idx) => {
        gridHtml += `
            <div class="relative group/item overflow-hidden rounded-lg border border-slate-800 bg-slate-900 aspect-square">
                <img src="${imgUri}" alt="Face #${idx+1}" class="w-full h-full object-cover ${renderingClass}">
                <div class="absolute inset-0 bg-slate-950/75 opacity-0 group-hover/item:opacity-100 transition-opacity duration-200 flex items-center justify-center gap-1 p-1">
                    <button onclick="openItemLightbox('${imgUri}', ${seed+idx})" class="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-white text-[10px] font-medium shadow-md">
                        Zoom
                    </button>
                    <button onclick="downloadSingleFace('${imgUri}', ${idx+1})" class="p-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-medium shadow-md">
                        Download
                    </button>
                </div>
            </div>
        `;
    });
    gridHtml += `</div>`;
    
    container.innerHTML = gridHtml;
    const captionEl = document.getElementById('preview-image-caption');
    if (captionEl) captionEl.textContent = `Generated Batch (${count} Faces) • Seed #${seed}`;
}

// RENDER SESSION GALLERY GRID
function renderGallery() {
    const galleryGrid = document.getElementById('gallery-grid');
    const emptyState = document.getElementById('gallery-empty-state');
    const countBadge = document.getElementById('gallery-count-badge');

    if (!galleryGrid || !emptyState) return;

    if (gallerySessionState.length === 0) {
        galleryGrid.classList.add('hidden');
        emptyState.classList.remove('hidden');
        if (countBadge) countBadge.textContent = `0 Faces`;
        return;
    }

    emptyState.classList.add('hidden');
    galleryGrid.classList.remove('hidden');
    if (countBadge) countBadge.textContent = `${gallerySessionState.length} Face${gallerySessionState.length > 1 ? 's' : ''}`;

    let html = '';
    const renderingClass = isCrispPixels ? 'crisp-pixel-rendering' : 'smooth-pixel-rendering';

    gallerySessionState.forEach((item, idx) => {
        const itemNum = idx + 1;

        html += `
        <div id="gallery-item-${item.id}" class="bg-slate-900/80 border border-slate-800/90 rounded-xl overflow-hidden p-2.5 space-y-2 group hover:border-indigo-500/50 transition-all duration-300 shadow-md">
            <div class="relative overflow-hidden rounded-lg aspect-square bg-slate-950 flex items-center justify-center cursor-pointer" onclick="openItemLightbox('${item.imageUri}', ${item.seed})">
                <img src="${item.imageUri}" alt="Face #${itemNum}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 ${renderingClass}">
                <div class="absolute inset-0 bg-slate-950/75 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-1.5 p-2">
                    <button onclick="event.stopPropagation(); openItemLightbox('${item.imageUri}', ${item.seed})" title="Inspect Enlarged View" class="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium shadow-lg transition-all flex items-center gap-1">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path></svg>
                    </button>
                    <button onclick="event.stopPropagation(); downloadSingleFace('${item.imageUri}', ${itemNum})" title="Download gan_face_${String(itemNum).padStart(3, '0')}.png" class="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium shadow-lg transition-all flex items-center gap-1">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    </button>
                    <button onclick="event.stopPropagation(); regenerateSingleItem('${item.id}')" title="Regenerate Item" class="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-amber-300 border border-slate-700 text-xs font-medium shadow-lg transition-all flex items-center gap-1">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                    </button>
                </div>
            </div>
            
            <div class="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
                <span class="text-indigo-400 font-semibold truncate">gan_face_${String(itemNum).padStart(3, '0')}.png</span>
                <div class="flex items-center gap-1.5 shrink-0">
                    <button onclick="regenerateSingleItem('${item.id}')" class="text-amber-400 hover:text-amber-300 text-[10px] font-sans underline">
                        Regen
                    </button>
                    <button onclick="downloadSingleFace('${item.imageUri}', ${itemNum})" class="text-slate-300 hover:text-white text-[10px] font-sans underline">
                        Save
                    </button>
                </div>
            </div>
        </div>
        `;
    });

    galleryGrid.innerHTML = html;
}

// RENDER SKELETON LOADERS
function renderGallerySkeletons(count) {
    const galleryGrid = document.getElementById('gallery-grid');
    const emptyState = document.getElementById('gallery-empty-state');
    if (!galleryGrid || !emptyState) return;

    emptyState.classList.add('hidden');
    galleryGrid.classList.remove('hidden');

    let skelHtml = '';
    for (let i = 0; i < count; i++) {
        skelHtml += `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-2.5 space-y-2 animate-pulse">
            <div class="aspect-square bg-slate-800/80 rounded-lg flex items-center justify-center skeleton-shimmer">
                <div class="w-6 h-6 rounded-full border-2 border-indigo-500/20 border-t-indigo-500 animate-spin"></div>
            </div>
            <div class="h-3 bg-slate-800 rounded w-2/3"></div>
        </div>
        `;
    }

    if (gallerySessionState.length > 0) {
        galleryGrid.innerHTML += skelHtml;
    } else {
        galleryGrid.innerHTML = skelHtml;
    }
}

// UI STATE MANAGEMENT
function hidePlaceholder() { 
    const el = document.getElementById('state-placeholder');
    if (el) { el.classList.add('hidden'); el.classList.remove('flex'); }
}
function showImagePreview() { 
    const el = document.getElementById('state-image');
    if (el) { el.classList.remove('hidden'); el.classList.add('flex'); }
}
function hideImagePreview() { 
    const el = document.getElementById('state-image');
    if (el) { el.classList.add('hidden'); el.classList.remove('flex'); }
}
function showLoading() { 
    const el = document.getElementById('state-loading');
    if (el) { el.classList.remove('hidden'); el.classList.add('flex'); }
}
function hideLoading() { 
    const el = document.getElementById('state-loading');
    if (el) { el.classList.add('hidden'); el.classList.remove('flex'); }
}

function hideError() {
    const errBox = document.getElementById('state-error');
    if (errBox) {
        errBox.classList.add('hidden');
        errBox.classList.remove('flex');
    }
}

function showError(msg) {
    const textEl = document.getElementById('error-message-text');
    const errBox = document.getElementById('state-error');
    if (textEl) textEl.textContent = msg;
    if (errBox) {
        errBox.classList.remove('hidden');
        errBox.classList.add('flex');
    }
    updateStatusBadge("Status: Error", "rose");
}

function clearError() {
    hideError();
    updateStatusBadge("Status: Ready", "slate");
    generateFaces(activeCount);
}

function updateStatusBadge(text, color) {
    const badge = document.getElementById('preview-badge');
    const dot = document.getElementById('preview-status-dot');
    if (badge) badge.textContent = text;
    if (dot) {
        dot.className = color === 'emerald' ? "w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" : (color === 'rose' ? "w-2.5 h-2.5 rounded-full bg-rose-500" : "w-2.5 h-2.5 rounded-full bg-slate-500");
    }
}
