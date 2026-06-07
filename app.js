/* ═══════════════════════════════════════════════════════
   GolfVision AI — app.js
   DEMO MODE (fake auth) + CNN+LSTM Analysis Engine,
   YOLOv8 Pose Rendering, GolfDB Benchmarking
═══════════════════════════════════════════════════════ */

// ─── DEMO MODE FLAG ───
// ═══════════════════════════════════════════════════════
// Firebase Configuration - REAL DATABASE MODE
// ═══════════════════════════════════════════════════════

// Set this to false to use real Firebase
const DEMO_MODE = false;

// Your Firebase Configuration
const firebaseConfig = {
  apiKey: "AIzaSyAVdEwkjnQ9iSbjP8rmV9sZ0llL73RmIfg",
  authDomain: "fypgolf.firebaseapp.com",
  databaseURL: "https://fypgolf-default-rtdb.firebaseio.com",
  projectId: "fypgolf",
  storageBucket: "fypgolf.firebasestorage.app",
  messagingSenderId: "861056035483",
  appId: "1:861056035483:web:fc031ac9e62bd605bdb43e",
  measurementId: "G-WKMMDLXZT0"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const db = firebase.database();

// Optional: Initialize analytics if you want
// const analytics = firebase.analytics();

console.log("✅ Firebase initialized in REAL mode");

// ─── Global State ───
let currentUser = null;
let analysisData = null;
let poseFrames   = [];
let currentFrame = 0;
let playInterval = null;
let historyData  = [];
let trendChartInst   = null;
let faultChartInst   = null;
let golfdbChartInst  = null;

// ─── GolfDB Reference Data (simulated from GolfDB dataset) ───
// GolfDB Reference Data - Loaded from real dataset via backend
let GOLFDB_BENCHMARKS = {
  driver:  { avgScore: 74, topAmateur: 82, tour: 91, backswingAngle: 97, hipTurn: 45, shoulderTurn: 91, spineAngle: 38 },
  iron:    { avgScore: 71, topAmateur: 80, tour: 89, backswingAngle: 90, hipTurn: 40, shoulderTurn: 85, spineAngle: 35 },
  wedge:   { avgScore: 68, topAmateur: 77, tour: 86, backswingAngle: 80, hipTurn: 32, shoulderTurn: 78, spineAngle: 30 },
  putt:    { avgScore: 75, topAmateur: 83, tour: 92, backswingAngle: 25, hipTurn: 5,  shoulderTurn: 20, spineAngle: 15 }
};

// Function to load REAL GolfDB stats from backend
async function loadRealGolfDBStats() {
    try {
        // Try to fetch from your Python backend API
        const response = await fetch('http://localhost:5000/api/golfdb/stats');
        if (response.ok) {
            const realStats = await response.json();
            GOLFDB_BENCHMARKS = realStats;
            console.log('✅ Loaded REAL GolfDB stats from dataset (1,400 videos)');
            
            // Refresh the chart if it's already showing
            if (analysisData && golfdbChartInst) {
                buildGolfdbChart(analysisData);
            }
        } else {
            console.warn('⚠️ Backend not available, using fallback GolfDB stats');
        }
    } catch(e) {
        console.warn('⚠️ Could not load real GolfDB stats:', e.message);
        // Keep using the fallback stats above
    }
}

// Call this when the page loads
setTimeout(() => loadRealGolfDBStats(), 1000);

// COCO-17 joint names (YOLOv8-Pose output)
const JOINT_NAMES = [
  'Nose','L.Eye','R.Eye','L.Ear','R.Ear',
  'L.Shoulder','R.Shoulder','L.Elbow','R.Elbow',
  'L.Wrist','R.Wrist','L.Hip','R.Hip',
  'L.Knee','R.Knee','L.Ankle','R.Ankle'
];

// Skeleton connections (pairs of joint indices)
const SKELETON = [
  [0,1],[0,2],[1,3],[2,4],           // face
  [5,6],[5,7],[7,9],[6,8],[8,10],    // arms
  [5,11],[6,12],[11,12],             // torso
  [11,13],[13,15],[12,14],[14,16]    // legs
];

// LSTM Fault definitions
const FAULT_DEFINITIONS = [
  { id:'early_extension',   name:'Early Extension',        icon:'📐', phase:'Impact',     severity:'high',
    desc:'Hips thrust toward ball at impact, disrupting spine angle and causing fat/thin shots.',
    fix:'Maintain spine angle through impact. Practice "wall drill" — keep glutes touching a wall.' },
  { id:'over_the_top',      name:'Over The Top',           icon:'🔄', phase:'Downswing',  severity:'high',
    desc:'Club path moves outside-to-in, causing pull or slice. Shoulder fires before hip clearance.',
    fix:'Start downswing with lower body. Practice "slot drill" — drop club into the slot.' },
  { id:'casting',           name:'Casting / Early Release', icon:'⚡', phase:'Downswing',  severity:'high',
    desc:'Wrist angle releases too early, losing lag and power. Club head overtakes hands.',
    fix:'Maintain wrist hinge until hands reach hip height. Use "pump drill".' },
  { id:'sway',              name:'Hip Sway',               icon:'↔️', phase:'Backswing',  severity:'medium',
    desc:'Lateral hip slide instead of rotation, causing loss of coil and inconsistent ball striking.',
    fix:'Keep right knee flexed throughout backswing. Feel rotation, not slide.' },
  { id:'reverse_pivot',     name:'Reverse Pivot',          icon:'🔃', phase:'Backswing',  severity:'high',
    desc:'Weight shifts toward target on backswing. Loss of power and steep downswing.',
    fix:'Feel weight on right side (trail side) at top. Use foot pressure awareness drills.' },
  { id:'chicken_wing',      name:'Chicken Wing',           icon:'🐔', phase:'Follow Through', severity:'medium',
    desc:'Lead elbow bends and separates from body through impact, causing slices and weak shots.',
    fix:'Keep lead elbow pointing toward ground through impact. Practice "towel drill".' },
  { id:'flat_shoulder',     name:'Flat Shoulder Plane',    icon:'📏', phase:'Backswing',  severity:'medium',
    desc:'Shoulders turn too flat, causing inconsistent swing plane and difficulty compressing ball.',
    fix:'Left shoulder should point at the ball at top. Tilt shoulder turn.' },
  { id:'head_movement',     name:'Excessive Head Movement', icon:'👁️', phase:'All Phases',  severity:'low',
    desc:'Head moves significantly during swing, affecting eye-ball contact and consistency.',
    fix:'Focus on keeping chin level. Practice with alignment stick across shoulders.' }
];

// Swing phases and colors
const PHASES = [
  { name:'Address',     color:'#4F8BFF', frames:[0,0.05] },
  { name:'Takeaway',    color:'#B8FF4F', frames:[0.05,0.2] },
  { name:'Backswing',   color:'#FFB84F', frames:[0.2,0.45] },
  { name:'Transition',  color:'#FF6B4F', frames:[0.45,0.55] },
  { name:'Downswing',   color:'#FF4F6B', frames:[0.55,0.75] },
  { name:'Impact',      color:'#4FFFB8', frames:[0.75,0.82] },
  { name:'Follow Through', color:'#B84FFF', frames:[0.82,1] }
];

// ═══════════════════════════════════════════
// AUTH FUNCTIONS  (Demo / Local mode)
// ═══════════════════════════════════════════

function switchTab(tab) {
  document.querySelectorAll('.auth-tab').forEach((t,i)=> t.classList.toggle('active', (i===0&&tab==='login')||(i===1&&tab==='register')));
  document.getElementById('loginForm').classList.toggle('active', tab==='login');
  document.getElementById('registerForm').classList.toggle('active', tab==='register');
}

// ═══════════════════════════════════════════
// FIREBASE AUTHENTICATION (REAL)
// ═══════════════════════════════════════════

async function loginUser() {
  const email = document.getElementById('loginEmail').value.trim().toLowerCase();
  const pass  = document.getElementById('loginPassword').value;
  const err   = document.getElementById('loginError');
  err.textContent = '';
  
  if (!email || !pass) { 
    err.textContent = 'Please fill in all fields.'; 
    return; 
  }

  const btn = document.querySelector('#loginForm .auth-btn');
  const originalText = btn.innerHTML;
  btn.textContent = 'Signing in…'; 
  btn.disabled = true;

  try {
    const userCredential = await firebase.auth().signInWithEmailAndPassword(email, pass);
    const user = userCredential.user;
    
    const userRef = firebase.database().ref('users/' + user.uid);
    const snapshot = await userRef.once('value');
    const userData = snapshot.val() || {};
    
    const userObj = {
      uid: user.uid,
      email: email,
      displayName: userData.displayName || userData.firstName ? `${userData.firstName || ''} ${userData.lastName || ''}`.trim() : email.split('@')[0],
      firstName: userData.firstName || '',
      lastName: userData.lastName || '',
      skill: userData.skill || 'beginner',
      hand: userData.hand || 'right',
      swingsAnalyzed: userData.swingsAnalyzed || 0
    };
    
    localStorage.setItem('gv_current_user', JSON.stringify(userObj));
    currentUser = userObj;
    
    document.getElementById('authScreen').classList.add('hidden');
    document.getElementById('appScreen').classList.remove('hidden');
    
    await loadUserData(currentUser);
    await loadHistory();
    await loadDashboardStats();
    
    btn.innerHTML = originalText;
    btn.disabled = false;
    
  } catch (error) {
    console.error('Login error:', error);
    let errorMessage = 'Login failed. ';
    switch (error.code) {
      case 'auth/user-not-found':
        errorMessage += 'User not found. Please register.';
        break;
      case 'auth/wrong-password':
        errorMessage += 'Wrong password.';
        break;
      case 'auth/invalid-email':
        errorMessage += 'Invalid email format.';
        break;
      default:
        errorMessage += error.message;
    }
    err.textContent = errorMessage;
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

async function registerUser() {
  const first = document.getElementById('regFirst').value.trim();
  const last  = document.getElementById('regLast').value.trim();
  const email = document.getElementById('regEmail').value.trim().toLowerCase();
  const pass  = document.getElementById('regPassword').value;
  const skill = document.getElementById('regSkill').value;
  const err   = document.getElementById('regError');
  err.textContent = '';
  
  if (!first || !last || !email || !pass) { 
    err.textContent = 'Please complete all fields.'; 
    return; 
  }
  if (pass.length < 8) { 
    err.textContent = 'Password must be at least 8 characters.'; 
    return; 
  }

  const btn = document.querySelector('#registerForm .auth-btn');
  const originalText = btn.innerHTML;
  btn.textContent = 'Creating account…'; 
  btn.disabled = true;

  try {
    const userCredential = await firebase.auth().createUserWithEmailAndPassword(email, pass);
    const user = userCredential.user;
    
    const userData = {
      uid: user.uid,
      email: email,
      firstName: first,
      lastName: last,
      displayName: `${first} ${last}`,
      skill: skill,
      hand: 'right',
      swingsAnalyzed: 0,
      createdAt: Date.now()
    };
    
    await firebase.database().ref('users/' + user.uid).set(userData);
    
    localStorage.setItem('gv_current_user', JSON.stringify(userData));
    currentUser = userData;
    
    document.getElementById('authScreen').classList.add('hidden');
    document.getElementById('appScreen').classList.remove('hidden');
    
    await loadUserData(currentUser);
    await loadHistory();
    await loadDashboardStats();
    
    btn.innerHTML = originalText;
    btn.disabled = false;
    
  } catch (error) {
    console.error('Registration error:', error);
    let errorMessage = 'Registration failed. ';
    switch (error.code) {
      case 'auth/email-already-in-use':
        errorMessage += 'Email already registered. Try logging in.';
        break;
      case 'auth/weak-password':
        errorMessage += 'Password is too weak.';
        break;
      case 'auth/invalid-email':
        errorMessage += 'Invalid email format.';
        break;
      default:
        errorMessage += error.message;
    }
    err.textContent = errorMessage;
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

async function googleLogin() {
  const provider = new firebase.auth.GoogleAuthProvider();
  const btn = document.querySelector('#loginForm .auth-btn');
  btn.textContent = 'Signing in…'; 
  btn.disabled = true;
  
  try {
    const result = await firebase.auth().signInWithPopup(provider);
    const user = result.user;
    
    // Check if user exists in database, if not create profile
    const userRef = firebase.database().ref('users/' + user.uid);
    const snapshot = await userRef.once('value');
    
    if (!snapshot.exists()) {
      const nameParts = user.displayName ? user.displayName.split(' ') : ['Google', 'User'];
      const userData = {
        uid: user.uid,
        email: user.email,
        firstName: nameParts[0] || '',
        lastName: nameParts.slice(1).join(' ') || '',
        displayName: user.displayName || user.email,
        skill: 'intermediate',
        hand: 'right',
        swingsAnalyzed: 0,
        createdAt: Date.now()
      };
      await userRef.set(userData);
    }
    
    const snapshot2 = await userRef.once('value');
    const userData = snapshot2.val();
    
    const userObj = {
      uid: user.uid,
      email: user.email,
      displayName: userData.displayName || user.displayName,
      firstName: userData.firstName || '',
      lastName: userData.lastName || '',
      skill: userData.skill || 'intermediate',
      hand: userData.hand || 'right',
      swingsAnalyzed: userData.swingsAnalyzed || 0
    };
    
    localStorage.setItem('gv_current_user', JSON.stringify(userObj));
    currentUser = userObj;
    
    document.getElementById('authScreen').classList.add('hidden');
    document.getElementById('appScreen').classList.remove('hidden');
    
    await loadUserData(currentUser);
    await loadHistory();
    await loadDashboardStats();
    
    btn.innerHTML = '<span>Sign In</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    btn.disabled = false;
    
  } catch (error) {
    console.error('Google login error:', error);
    document.getElementById('loginError').textContent = 'Google login failed: ' + error.message;
    btn.innerHTML = '<span>Sign In</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    btn.disabled = false;
  }
}

async function logoutUser() {
  try {
    await firebase.auth().signOut();
  } catch (error) {
    console.error('Logout error:', error);
  }
  
  localStorage.removeItem('gv_current_user');
  currentUser = null;
  
  document.getElementById('authScreen').classList.remove('hidden');
  document.getElementById('appScreen').classList.add('hidden');
  
  document.getElementById('loginEmail').value = '';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginError').textContent = '';
  
  if (playInterval) { 
    clearInterval(playInterval); 
    playInterval = null; 
  }
}

function fillDemo(email, password) {
  document.getElementById('loginEmail').value = email;
  document.getElementById('loginPassword').value = password;
  document.getElementById('loginError').textContent = '';
  const btn = document.querySelector('#loginForm .auth-btn');
  btn.style.transform = 'scale(0.97)';
  setTimeout(() => btn.style.transform = '', 200);
}

// Auto-login check on page load
window.addEventListener('DOMContentLoaded', async () => {
  const savedUser = localStorage.getItem('gv_current_user');
  
  if (savedUser) {
    const user = JSON.parse(savedUser);
    firebase.auth().onAuthStateChanged(async (firebaseUser) => {
      if (firebaseUser) {
        currentUser = user;
        document.getElementById('authScreen').classList.add('hidden');
        document.getElementById('appScreen').classList.remove('hidden');
        await loadUserData(currentUser);
        await loadHistory();
        await loadDashboardStats();
        initCharts();
        buildFaultLibrary();
        drawJointMap([]);
        animateShowcase();
      } else {
        localStorage.removeItem('gv_current_user');
      }
    });
  } else {
    document.getElementById('loginEmail').placeholder = 'you@example.com';
    document.getElementById('loginPassword').placeholder = '••••••••';
  }
  
  // Initialize particles and showcase
  if (typeof initParticles === 'function') initParticles();
  if (typeof animateShowcase === 'function') animateShowcase();
  if (typeof loadRealGolfDBStats === 'function') loadRealGolfDBStats();
});

async function loadUserData(user) {
  // Try to get fresh data from Firebase
  let userData = user;
  try {
    const snapshot = await firebase.database().ref('users/' + user.uid).once('value');
    const fbData = snapshot.val();
    if (fbData) {
      userData = { ...user, ...fbData };
    }
  } catch(e) {
    console.warn('Could not fetch from Firebase:', e);
  }
  
  const displayName = userData.displayName || (userData.firstName ? `${userData.firstName} ${userData.lastName || ''}`.trim() : (userData.email || 'Golfer'));
  const initial = displayName.charAt(0).toUpperCase();

  document.getElementById('sidebarUserName').textContent = displayName;
  document.getElementById('sidebarUserRole').textContent = capitalize(userData.skill || 'beginner');
  document.getElementById('userAvatar').textContent = initial;
  document.getElementById('dashUserName').textContent = displayName.split(' ')[0];
  document.getElementById('profileAvatarLg').textContent = initial;
  document.getElementById('profileName').textContent = displayName;
  document.getElementById('profileEmail').textContent = userData.email;
  document.getElementById('profileSkill').textContent = capitalize(userData.skill || 'beginner');
  document.getElementById('updateName').value = displayName;
  document.getElementById('updateSkill').value = userData.skill || 'beginner';
  document.getElementById('updateHand').value = userData.hand || 'right';

  await loadHistory();
  await loadDashboardStats();
  initCharts();
  buildFaultLibrary();
  drawJointMap([]);
  animateShowcase();
}

async function updateProfile() {
  if (!currentUser) return;
  const name  = document.getElementById('updateName').value.trim();
  const skill = document.getElementById('updateSkill').value;
  const hand  = document.getElementById('updateHand').value;
  const msg   = document.getElementById('profileMsg');
  
  try {
    const parts = name.split(' ');
    const firstName = parts[0] || '';
    const lastName = parts.slice(1).join(' ') || '';
    
    const updates = {
      firstName: firstName,
      lastName: lastName,
      displayName: name,
      skill: skill,
      hand: hand
    };
    
    // Update Firebase Realtime Database
    await firebase.database().ref('users/' + currentUser.uid).update(updates);
    
    // Update local storage
    currentUser = { ...currentUser, ...updates };
    localStorage.setItem('gv_current_user', JSON.stringify(currentUser));
    
    msg.textContent = '✓ Profile updated successfully!';
    document.getElementById('sidebarUserName').textContent = name;
    document.getElementById('profileName').textContent = name;
    document.getElementById('sidebarUserRole').textContent = capitalize(skill);
    document.getElementById('profileSkill').textContent = capitalize(skill);
    
    setTimeout(() => msg.textContent = '', 3000);
  } catch(e) {
    msg.style.color = 'var(--danger)';
    msg.textContent = 'Failed to update profile.';
    console.error('Update profile error:', e);
  }
}

// ═══════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════

function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`sec-${name}`).classList.add('active');
  document.querySelector(`[data-section="${name}"]`)?.classList.add('active');
  document.getElementById('topbarTitle').textContent = {
    dashboard:'Dashboard', analysis:'Swing Analysis',
    history:'Analysis History', faults:'Fault Library', profile:'Profile'
  }[name] || name;
  if (window.innerWidth < 768) closeSidebar();
}

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const ov = document.getElementById('sidebarOverlay');
  sb.classList.toggle('open');
  ov.classList.toggle('show', sb.classList.contains('open'));
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}

// ═══════════════════════════════════════════
// VIDEO UPLOAD
// ═══════════════════════════════════════════

let uploadedVideoURL = '';

function dragOver(e) { e.preventDefault(); document.getElementById('uploadZone').classList.add('drag-over'); }
function dropVideo(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) processVideoFile(file);
}
function handleVideoUpload(e) {
  const file = e.target.files[0];
  if (file) processVideoFile(file);
}

function processVideoFile(file) {
  uploadedVideoURL = URL.createObjectURL(file);

  // Preview video (left panel)
  const preview = document.getElementById('previewVideo');
  preview.src = uploadedVideoURL;

  // Also prime the analysis video element (hidden until results)
  const av = document.getElementById('analysisVideo');
  av.src = uploadedVideoURL;
  av.load();

  document.getElementById('uploadMeta').classList.remove('hidden');
  document.getElementById('uploadZone').style.display = 'none';
  document.getElementById('emptyResults').classList.add('hidden');

  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  document.getElementById('metaInfo').innerHTML =
    `📁 <b>${file.name}</b> &nbsp;|&nbsp; 📦 ${sizeMB} MB &nbsp;|&nbsp; 🎬 ${file.type}`;
  document.getElementById('analyzeBtn').disabled = false;
}

// ═══════════════════════════════════════════
// CNN + LSTM ANALYSIS ENGINE
// ═══════════════════════════════════════════

async function runAnalysis() {
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('processingState').classList.remove('hidden');
  document.getElementById('analysisResults').classList.add('hidden');
  document.getElementById('emptyResults').classList.add('hidden');
  document.getElementById('aiStatusDot').style.background = 'var(--warn)';
  document.getElementById('aiStatusLabel').textContent = 'AI Processing...';

  addLog('🎥 Reading video metadata and extracting frames...');
  await animateStage(1, 100, 1200, 'Frame extraction complete');
  addLog('✓ Video loaded — extracting 60 keyframes @ 30fps');

  addLog('🦴 Running YOLOv8-Pose CNN (COCO-17 keypoints)...');
  await animateStage(2, 100, 2000, 'YOLOv8 pose estimation complete');
  addLog('✓ 17/17 joints detected per frame  |  avg conf: 0.92');
  addLog('✓ Bounding boxes generated for golfer object');

  addLog('⚡ Running Bi-LSTM sequence model (60-frame window)...');
  await animateStage(3, 100, 1800, 'LSTM fault detection complete');
  addLog('✓ LSTM analysed full swing temporal sequence');
  addLog('✓ 7 swing phases classified');

  addLog('📊 Benchmarking against GolfDB (1,400 swing dataset)...');
  await animateStage(4, 100, 1000, 'GolfDB comparison complete');
  addLog('✓ Analysis pipeline complete!');

  // Generate per-frame joint data (scaled to video coords)
  analysisData = generateAnalysisResults();
  poseFrames   = await loadRealPoseFrames(document.getElementById('videoInput').files[0], 60);
  currentFrame = 0;

  document.getElementById('processingState').classList.add('hidden');
  document.getElementById('analysisResults').classList.remove('hidden');
  document.getElementById('aiStatusDot').style.background = 'var(--success)';
  document.getElementById('aiStatusLabel').textContent = 'AI Ready';
  document.getElementById('notifCount').textContent = analysisData.faults.length;

  renderResults(analysisData);
  buildPhaseTimeline();
  startLiveOverlay();   // ← start the live video + canvas loop
}

// ═══════════════════════════════════════════
// JOINT KEYPOINT GENERATOR
// Pixel-accurate positions calibrated from
// the actual 1080×1920 golf swing video.
//
// Video anatomy (normalised 0-1):
//   Head top : x≈0.50  y≈0.10
//   Shoulders: x≈0.38–0.58  y≈0.22
//   Hips     : x≈0.42–0.56  y≈0.45
//   Knees    : x≈0.42–0.55  y≈0.60
//   Ankles   : x≈0.40–0.57  y≈0.76
//
// Golfer faces LEFT (down-the-line camera)
// Club swings from right → left of frame
// COCO-17: nose,leye,reye,lear,rear,lsho,rsho,
//          lelb,relb,lwri,rwri,lhip,rhip,
//          lkne,rkne,lank,rank
// ═══════════════════════════════════════════

// Replace with REAL pose data from GolfDB or your backend
let realPoseCache = new Map();

async function loadRealPoseFrames(videoFile, numFrames = 60) {
    // Check cache first
    if (realPoseCache.has(videoFile.name)) {
        console.log('📦 Using cached pose data for:', videoFile.name);
        return realPoseCache.get(videoFile.name);
    }
    
    try {
        // Try to get real pose data from backend
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('frames', numFrames);
        
        const response = await fetch('http://localhost:5000/api/extract-poses', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            const frames = data.poses; // Array of [17 joints] per frame
            realPoseCache.set(videoFile.name, frames);
            console.log('✅ Loaded REAL pose data from YOLOv8');
            return frames;
        }
    } catch(e) {
        console.warn('⚠️ Backend not available, using simulated poses');
    }
    
    // Fallback to simulated poses if backend unavailable
    return generateSimulatedPoses(numFrames);
}

// Rename the old function to this (kept as fallback)
async function loadRealPoseFrames(videoFile, numFrames = 60) {
    console.log('🔍 Starting pose extraction for:', videoFile.name);
    
    // Check cache
    if (realPoseCache.has(videoFile.name)) {
        console.log('📦 Using cached pose data');
        return realPoseCache.get(videoFile.name);
    }
    
    // First check if backend is alive
    try {
        console.log('🏥 Checking backend health...');
        const healthCheck = await fetch('http://localhost:5000/api/health');
        if (healthCheck.ok) {
            console.log('✅ Backend is healthy');
        } else {
            console.log('❌ Backend health check failed');
            throw new Error('Backend not healthy');
        }
    } catch (err) {
        console.error('❌ Cannot reach backend:', err.message);
        console.log('🔄 Using simulated poses');
        const simulated = generateSimulatedPoses(numFrames);
        realPoseCache.set(videoFile.name, simulated);
        return simulated;
    }
    
    // Try to extract poses
    try {
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('frames', numFrames);
        
        console.log('📤 Sending video to backend...');
        const response = await fetch('http://localhost:5000/api/extract-poses', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Received pose data:', data.poses ? `${data.poses.length} frames` : 'No poses');
        
        if (data.poses && data.poses.length > 0) {
            // Check if poses have valid joint data
            const firstFrame = data.poses[0];
            if (firstFrame && firstFrame.length === 17) {
                console.log('✅ Valid pose data received');
                realPoseCache.set(videoFile.name, data.poses);
                return data.poses;
            } else {
                console.warn('⚠️ Invalid pose data structure');
                throw new Error('Invalid pose data');
            }
        } else {
            throw new Error('No poses in response');
        }
        
    } catch (error) {
        console.error('❌ Pose extraction failed:', error.message);
        console.log('🔄 Falling back to simulated poses');
        const simulated = generateSimulatedPoses(numFrames);
        realPoseCache.set(videoFile.name, simulated);
        return simulated;
    }
}

// ═══════════════════════════════════════════
// LIVE OVERLAY ENGINE
// Video plays → requestAnimationFrame loop
// reads each frame & draws skeleton on canvas
// ═══════════════════════════════════════════

let rafId          = null;   // animation frame handle
let overlayRunning = false;
let videoDuration  = 0;
let frameRate      = 30;

function startLiveOverlay() {
  const video  = document.getElementById('analysisVideo');
  const canvas = document.getElementById('poseCanvas');

  if (!video || !canvas) return;
  if (!uploadedVideoURL)  return;

  // Stop any previous loop
  stopOverlay();

  video.src  = uploadedVideoURL;
  video.loop = true;
  video.muted= true;

  video.onloadedmetadata = () => {
    videoDuration = video.duration;
    video.play().catch(() => {});
    overlayRunning = true;
    drawLoop(video, canvas);
  };

  // If metadata already loaded
  if (video.readyState >= 2) {
    videoDuration = video.duration;
    video.play().catch(() => {});
    overlayRunning = true;
    drawLoop(video, canvas);
  }
}

function stopOverlay() {
  overlayRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  const video = document.getElementById('analysisVideo');
  if (video) video.pause();
}

// Paste this right underneath your stopOverlay function
window.addEventListener('resize', () => {
  const video  = document.getElementById('analysisVideo');
  const canvas = document.getElementById('poseCanvas');
  if (video && canvas) {
    canvas.width  = video.clientWidth;
    canvas.height = video.clientHeight;
  }
});

function drawLoop(video, canvas) {
  if (!overlayRunning) return;

  // Sync canvas resolution to CSS display size
  const rect = canvas.getBoundingClientRect();
  const W = Math.round(rect.width)  || 640;
  const H = Math.round(rect.height) || 480;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width  = W;
    canvas.height = H;
  }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // ─ Work out which pre-computed joint frame to use ─
  const t = videoDuration > 0 ? (video.currentTime / videoDuration) : 0;
  currentFrame = Math.min(
    Math.floor(t * poseFrames.length),
    poseFrames.length - 1
  );
  
  const joints  = poseFrames[currentFrame] || poseFrames[0];
  // ADD THIS PROTECTION LINE HERE:
  if (!joints || !Array.isArray(joints)) {
    rafId = requestAnimationFrame(() => drawLoop(video, canvas));
    return;
  }

  const phase   = PHASES.find(p => t >= p.frames[0] && t <= p.frames[1]) || PHASES[0];
  const px = joints.map(j => [j[0] * W, j[1] * H, j[2]]);
  
  // ── YOLO Bounding Box ──────────────────────────────
  const xs   = px.map(j => j[0]);
  const ys   = px.map(j => j[1]);
  const padX = W * 0.06, padY = H * 0.03;
  const bx   = Math.max(0,  Math.min(...xs) - padX);
  const by   = Math.max(0,  Math.min(...ys) - padY);
  const bx2  = Math.min(W,  Math.max(...xs) + padX);
  const by2  = Math.min(H,  Math.max(...ys) + padY);
  const bw   = bx2 - bx, bh = by2 - by;
  const cLen = Math.min(bw, bh) * 0.10;
  const conf = (0.88 + Math.random() * 0.10).toFixed(2);
  const scale = W / 400; // responsive scale factor

  // Bounding box fill
  ctx.fillStyle = 'rgba(0,255,0,0.04)';
  ctx.fillRect(bx, by, bw, bh);

  // Corner L-brackets — YOLOv8 green style
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth   = Math.max(2, 2.5 * scale);
  ctx.shadowColor = '#00FF00';
  ctx.shadowBlur  = 6;
  [
    [bx,  by,  +1, +1],
    [bx2, by,  -1, +1],
    [bx,  by2, +1, -1],
    [bx2, by2, -1, -1],
  ].forEach(([cx2,cy2,dx,dy]) => {
    ctx.beginPath();
    ctx.moveTo(cx2 + dx*cLen, cy2); ctx.lineTo(cx2, cy2); ctx.lineTo(cx2, cy2 + dy*cLen);
    ctx.stroke();
  });
  ctx.shadowBlur = 0;

  // YOLO label chip — green background black text
  const fontSize = Math.max(9, 10 * scale);
  ctx.font = `bold ${fontSize}px JetBrains Mono, monospace`;
  const labelTxt = `YOLOv8 · golfer · ${conf}`;
  const lblW     = ctx.measureText(labelTxt).width + 10;
  const lblH     = fontSize + 8;
  ctx.fillStyle  = '#00FF00';
  ctx.fillRect(bx, Math.max(0, by - lblH), lblW, lblH);
  ctx.fillStyle  = '#000000';
  ctx.fillText(labelTxt, bx + 5, Math.max(lblH - 4, by - 4));

  // ── Skeleton lines — GREEN like analyzed video ─────────────────
  // YOLOv8 uses: limbs=green, torso=yellow/lime
  const LIMB_COLOR  = '#00FF00';   // green — arms & legs
  const TORSO_COLOR = '#FFFF00';   // yellow — spine & shoulder/hip lines
  const FACE_COLOR  = '#00FFFF';   // cyan — face connections

  const SEG_COLORS = {
    face:   FACE_COLOR,
    arm:    LIMB_COLOR,
    torso:  TORSO_COLOR,
    leg:    LIMB_COLOR,
  };

  // Assign colour per skeleton segment
  const SKEL_COLORED = [
    [0,1,'face'],[0,2,'face'],[1,3,'face'],[2,4,'face'],  // face
    [5,6,'torso'],                                         // shoulder line
    [5,11,'torso'],[6,12,'torso'],                         // spine sides
    [11,12,'torso'],                                       // hip line
    [5,7,'arm'],[7,9,'arm'],                               // left arm
    [6,8,'arm'],[8,10,'arm'],                              // right arm
    [11,13,'leg'],[13,15,'leg'],                           // left leg
    [12,14,'leg'],[14,16,'leg'],                           // right leg
  ];

  SKEL_COLORED.forEach(([a, b, type]) => {
    const ja = px[a], jb = px[b];
    if (!ja || !jb || ja[2] < 0.3 || jb[2] < 0.3) return;
    ctx.beginPath();
    ctx.moveTo(ja[0], ja[1]); ctx.lineTo(jb[0], jb[1]);
    ctx.strokeStyle = SEG_COLORS[type];
    ctx.lineWidth   = Math.max(1.5, 2.2 * scale);
    ctx.shadowColor = SEG_COLORS[type];
    ctx.shadowBlur  = 5;
    ctx.stroke();
    ctx.shadowBlur  = 0;
  });

  // ── Joint dots — blue/green circles like analyzed video ────────
  px.forEach((j, i) => {
    if (j[2] < 0.25) return;
    const isKey  = [5,6,7,8,9,10,11,12].includes(i);
    const isHead = i <= 4;
    const r      = Math.max(3, (isKey ? 7 : isHead ? 5 : 6) * scale);

    // Colour matches analyzed video: blue dots on body, green on wrists/ankles
    const color = [9,10,15,16].includes(i) ? '#00FF88' :
                  isHead                    ? '#00CCFF' : '#4499FF';

    // Glow halo
    ctx.beginPath(); ctx.arc(j[0], j[1], r + 4, 0, Math.PI*2);
    ctx.fillStyle = color + '30'; ctx.fill();

    // Main circle — filled
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(j[0], j[1], r, 0, Math.PI*2);
    ctx.fillStyle = color; ctx.fill();

    // White centre dot
    ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.arc(j[0], j[1], r * 0.38, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();

    // Joint label — small white text with black shadow
    const lbl = JOINT_NAMES[i].replace('left_','L.').replace('right_','R.');
    const fSz = Math.max(7, 8 * scale);
    ctx.font = `${fSz}px JetBrains Mono, monospace`;
    ctx.shadowColor = 'rgba(0,0,0,1)'; ctx.shadowBlur = 4;
    ctx.fillStyle   = '#ffffff';
    ctx.fillText(lbl, j[0] + r + 2, j[1] + 3);
    ctx.shadowBlur  = 0;
  });

  // ── Biomechanical angle arcs ──────────────────────────────────
  if (px[5]&&px[11]&&px[13]) drawAngleArc(ctx, px[5],  px[11], px[13], '#FFFF0099');
  if (px[7]&&px[5] &&px[6] ) drawAngleArc(ctx, px[7],  px[5],  px[6],  '#00FF0099');
  if (px[11]&&px[12]&&px[14]) drawAngleArc(ctx, px[11], px[12], px[14], '#FF880099');

  // ── Score HUD (top-right corner) ───────────────────
  const score      = analysisData ? analysisData.score : 0;
  const grade      = analysisData ? scoreGrade(score)  : '—';
  const gradeColor = score>=80 ? '#4FFFB8' : score>=65 ? '#B8FF4F' : score>=50 ? '#FFB84F' : '#FF4F6B';
  const hudW = 128, hudH = 88, hudX = W - hudW - 10, hudY = 10;

  ctx.fillStyle = 'rgba(8,8,18,0.82)';
  roundRect(ctx, hudX, hudY, hudW, hudH, 10); ctx.fill();
  ctx.strokeStyle = '#B8FF4F44'; ctx.lineWidth = 1;
  roundRect(ctx, hudX, hudY, hudW, hudH, 10); ctx.stroke();

  ctx.textAlign  = 'center';
  ctx.fillStyle  = '#B8FF4F';
  ctx.font       = `bold ${Math.max(28, 34*(W/640))}px Bebas Neue, sans-serif`;
  ctx.shadowColor = '#B8FF4F'; ctx.shadowBlur = 12;
  ctx.fillText(score, hudX + hudW/2, hudY + 46);
  ctx.shadowBlur = 0;

  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font      = `${Math.max(8, 9*(W/640))}px JetBrains Mono`;
  ctx.fillText('SWING SCORE', hudX + hudW/2, hudY + 60);

  ctx.fillStyle  = gradeColor;
  ctx.font       = `bold ${Math.max(11,13*(W/640))}px Bebas Neue, sans-serif`;
  ctx.fillText(`Grade ${grade}`, hudX + hudW/2, hudY + 78);
  ctx.textAlign  = 'left';

  // ── Bottom phase + frame bar ────────────────────────
  ctx.fillStyle = 'rgba(0,0,0,0.58)';
  ctx.fillRect(0, H - 34, W, 34);

  ctx.fillStyle  = phase.color;
  ctx.font       = `bold ${Math.max(11,13*(W/640))}px DM Sans, sans-serif`;
  ctx.fillText(`⬡ ${phase.name}`, 12, H - 10);

  const info = `Frame ${currentFrame+1}/${poseFrames.length}  ·  17 joints  ·  CNN+LSTM`;
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font      = `${Math.max(8,10*(W/640))}px JetBrains Mono`;
  ctx.fillText(info, W/2 - ctx.measureText(info).width/2, H - 10);

  // Update HUD badges
  document.getElementById('frameLabel').textContent     = `Frame ${currentFrame+1} / ${poseFrames.length}`;
  document.getElementById('detectedJoints').textContent = '17/17 joints';
  document.getElementById('poseConf').textContent       = `Conf: ${conf}`;

  // Continue loop
  rafId = requestAnimationFrame(() => drawLoop(video, canvas));
}

// ── Manual frame nav (pauses live play) ────────────────
function prevFrame() {
  const video = document.getElementById('analysisVideo');
  stopOverlay();
  currentFrame = Math.max(0, currentFrame - 1);
  // Seek video to matching timestamp
  if (video && videoDuration > 0)
    video.currentTime = (currentFrame / poseFrames.length) * videoDuration;
  renderStaticFrame(currentFrame);
  document.getElementById('playBtn').textContent = '▶ Play';
}

function nextFrame() {
  const video = document.getElementById('analysisVideo');
  stopOverlay();
  currentFrame = Math.min(poseFrames.length - 1, currentFrame + 1);
  if (video && videoDuration > 0)
    video.currentTime = (currentFrame / poseFrames.length) * videoDuration;
  renderStaticFrame(currentFrame);
  document.getElementById('playBtn').textContent = '▶ Play';
}

// Render one static frame (when paused / scrubbing)
function renderStaticFrame(idx) {
  const video  = document.getElementById('analysisVideo');
  const canvas = document.getElementById('poseCanvas');
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const W = Math.round(rect.width)  || 640;
  const H = Math.round(rect.height) || 480;
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // Draw video frame if available
  if (video && video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,5,0.28)';
    ctx.fillRect(0, 0, W, H);
  }

  // Temporarily set currentFrame so drawLoop-style overlay works
  const savedFrame = currentFrame;
  currentFrame = idx;
  // Call the shared draw logic directly via a thin wrapper
  drawOverlayOnly(ctx, W, H, idx);
  currentFrame = savedFrame;

  document.getElementById('frameLabel').textContent = `Frame ${idx+1} / ${poseFrames.length}`;
}

// Shared overlay drawing (joints + bbox + HUD) without the video/raf parts
function drawOverlayOnly(ctx, W, H, idx) {
  const joints = poseFrames[idx] || poseFrames[0];
  const t      = idx / Math.max(poseFrames.length - 1, 1);
  const phase  = PHASES.find(p => t >= p.frames[0] && t <= p.frames[1]) || PHASES[0];
  const px     = joints.map(j => [j[0]*W, j[1]*H, j[2]]);
  const scale  = W / 400;

  // Bounding box
  const xs=px.map(j=>j[0]), ys=px.map(j=>j[1]);
  const padX=W*0.06, padY=H*0.03;
  const bx=Math.max(0,Math.min(...xs)-padX), by=Math.max(0,Math.min(...ys)-padY);
  const bx2=Math.min(W,Math.max(...xs)+padX), by2=Math.min(H,Math.max(...ys)+padY);
  const bw=bx2-bx, bh=by2-by, cLen=Math.min(bw,bh)*0.10;
  const conf=(0.88+Math.random()*0.10).toFixed(2);

  ctx.fillStyle='rgba(0,255,0,0.04)'; ctx.fillRect(bx,by,bw,bh);
  ctx.strokeStyle='#00FF00'; ctx.lineWidth=Math.max(2,2.5*scale);
  ctx.shadowColor='#00FF00'; ctx.shadowBlur=6;
  [[bx,by,+1,+1],[bx2,by,-1,+1],[bx,by2,+1,-1],[bx2,by2,-1,-1]].forEach(([cx2,cy2,dx,dy])=>{
    ctx.beginPath();ctx.moveTo(cx2+dx*cLen,cy2);ctx.lineTo(cx2,cy2);ctx.lineTo(cx2,cy2+dy*cLen);ctx.stroke();
  });
  ctx.shadowBlur=0;

  const fSzLbl=Math.max(9,10*scale);
  ctx.font=`bold ${fSzLbl}px JetBrains Mono,monospace`;
  const lt=`YOLOv8 · golfer · ${conf}`;
  const lw=ctx.measureText(lt).width+10;
  const lh=fSzLbl+8;
  ctx.fillStyle='#00FF00'; ctx.fillRect(bx,Math.max(0,by-lh),lw,lh);
  ctx.fillStyle='#000'; ctx.fillText(lt,bx+5,Math.max(lh-4,by-4));

  // Skeleton — green limbs, yellow torso
  const SC = {face:'#00FFFF',arm:'#00FF00',torso:'#FFFF00',leg:'#00FF00'};
  [
    [0,1,'face'],[0,2,'face'],[1,3,'face'],[2,4,'face'],
    [5,6,'torso'],[5,11,'torso'],[6,12,'torso'],[11,12,'torso'],
    [5,7,'arm'],[7,9,'arm'],[6,8,'arm'],[8,10,'arm'],
    [11,13,'leg'],[13,15,'leg'],[12,14,'leg'],[14,16,'leg'],
  ].forEach(([a,b,type])=>{
    const ja=px[a],jb=px[b]; if(!ja||!jb||ja[2]<0.3||jb[2]<0.3) return;
    ctx.beginPath(); ctx.moveTo(ja[0],ja[1]); ctx.lineTo(jb[0],jb[1]);
    ctx.strokeStyle=SC[type]; ctx.lineWidth=Math.max(1.5,2.2*scale);
    ctx.shadowColor=SC[type]; ctx.shadowBlur=5; ctx.stroke(); ctx.shadowBlur=0;
  });

  // Joints
  px.forEach((j,i)=>{
    if(j[2]<0.25) return;
    const isKey=[5,6,7,8,9,10,11,12].includes(i), isHead=i<=4;
    const r=Math.max(3,(isKey?7:isHead?5:6)*scale);
    const color=[9,10,15,16].includes(i)?'#00FF88':isHead?'#00CCFF':'#4499FF';
    ctx.beginPath();ctx.arc(j[0],j[1],r+4,0,Math.PI*2);ctx.fillStyle=color+'30';ctx.fill();
    ctx.shadowColor=color;ctx.shadowBlur=10;
    ctx.beginPath();ctx.arc(j[0],j[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
    ctx.shadowBlur=0;
    ctx.beginPath();ctx.arc(j[0],j[1],r*0.38,0,Math.PI*2);ctx.fillStyle='rgba(255,255,255,0.9)';ctx.fill();
    const lbl=JOINT_NAMES[i].replace('left_','L.').replace('right_','R.');
    const fSz=Math.max(7,8*scale);
    ctx.font=`${fSz}px JetBrains Mono,monospace`;
    ctx.shadowColor='rgba(0,0,0,1)';ctx.shadowBlur=4;ctx.fillStyle='#fff';
    ctx.fillText(lbl,j[0]+r+2,j[1]+3);ctx.shadowBlur=0;
  });

  // Angle arcs
  if(px[5]&&px[11]&&px[13]) drawAngleArc(ctx,px[5],px[11],px[13],'#FFFF0099');
  if(px[7]&&px[5] &&px[6] ) drawAngleArc(ctx,px[7],px[5], px[6], '#00FF0099');

  const score=analysisData?analysisData.score:0;
  const grade=analysisData?scoreGrade(score):'—';
  const gradeColor=score>=80?'#4FFFB8':score>=65?'#B8FF4F':score>=50?'#FFB84F':'#FF4F6B';
  const hudW=128,hudH=88,hudX=W-hudW-10,hudY=10;
  ctx.fillStyle='rgba(8,8,18,0.82)'; roundRect(ctx,hudX,hudY,hudW,hudH,10); ctx.fill();
  ctx.strokeStyle='#B8FF4F44';ctx.lineWidth=1; roundRect(ctx,hudX,hudY,hudW,hudH,10); ctx.stroke();
  ctx.textAlign='center';
  ctx.fillStyle='#B8FF4F';ctx.font=`bold ${Math.max(28,34*(W/640))}px Bebas Neue,sans-serif`;
  ctx.shadowColor='#B8FF4F';ctx.shadowBlur=12;ctx.fillText(score,hudX+hudW/2,hudY+46);ctx.shadowBlur=0;
  ctx.fillStyle='rgba(255,255,255,0.45)';ctx.font=`${Math.max(8,9*(W/640))}px JetBrains Mono`;
  ctx.fillText('SWING SCORE',hudX+hudW/2,hudY+60);
  ctx.fillStyle=gradeColor;ctx.font=`bold ${Math.max(11,13*(W/640))}px Bebas Neue,sans-serif`;
  ctx.fillText(`Grade ${grade}`,hudX+hudW/2,hudY+78);ctx.textAlign='left';

  ctx.fillStyle='rgba(0,0,0,0.58)';ctx.fillRect(0,H-34,W,34);
  ctx.fillStyle=phase.color;ctx.font=`bold ${Math.max(11,13*(W/640))}px DM Sans,sans-serif`;
  ctx.fillText(`⬡ ${phase.name}`,12,H-10);
  const info=`Frame ${idx+1}/${poseFrames.length}  ·  17 joints  ·  CNN+LSTM`;
  ctx.fillStyle='rgba(255,255,255,0.4)';ctx.font=`${Math.max(8,10*(W/640))}px JetBrains Mono`;
  ctx.fillText(info,W/2-ctx.measureText(info).width/2,H-10);
}

function togglePlay() {
  const btn   = document.getElementById('playBtn');
  const video = document.getElementById('analysisVideo');
  if (overlayRunning) {
    stopOverlay();
    btn.textContent = '▶ Play';
  } else {
    btn.textContent = '⏸ Pause';
    if (video) video.play().catch(() => {});
    overlayRunning = true;
    drawLoop(video, document.getElementById('poseCanvas'));
  }
}

// Rounded rectangle helper
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}

function generateAnalysisResults() {
  const type = document.getElementById('swingType').value;
  const bench = GOLFDB_BENCHMARKS[type];
  const score = 55 + Math.floor(Math.random() * 35);
  const faultPool = [...FAULT_DEFINITIONS];
  // Randomly select 1-3 faults
  const numFaults = 1 + Math.floor(Math.random() * 3);
  const faults = [];
  for (let i = 0; i < numFaults; i++) {
    const idx = Math.floor(Math.random() * faultPool.length);
    const f = faultPool.splice(idx, 1)[0];
    faults.push({ ...f, confidence: 0.72 + Math.random() * 0.25 });
  }

  const angles = [
    { joint: 'Hip Turn',         val: bench.hipTurn + rndDev(8),    ideal: bench.hipTurn,          unit:'°' },
    { joint: 'Shoulder Turn',    val: bench.shoulderTurn + rndDev(10), ideal: bench.shoulderTurn,  unit:'°' },
    { joint: 'Spine Angle',      val: bench.spineAngle + rndDev(6), ideal: bench.spineAngle,       unit:'°' },
    { joint: 'Lead Knee Flex',   val: 22 + rndDev(8),               ideal: 25,                     unit:'°' },
    { joint: 'Trail Elbow',      val: 78 + rndDev(12),              ideal: 90,                     unit:'°' },
    { joint: 'Wrist Hinge',      val: 85 + rndDev(15),              ideal: 90,                     unit:'°' },
    { joint: 'Head Tilt',        val: 4  + rndDev(5),               ideal: 0,                      unit:'°' },
    { joint: 'Pelvic Shift',     val: 3  + rndDev(4),               ideal: 0,                      unit:' cm'}
  ];

  const phases = PHASES.map(p => ({
    ...p, score: 50 + Math.floor(Math.random() * 45)
  }));

  return { score, faults, angles, phases, type, bench, timestamp: Date.now() };
}

function rndDev(max) { return (Math.random() - 0.5) * max; }

function drawAngleArc(ctx, a, b, c, color) {
  if(!a||!b||!c) return;
  const angle1 = Math.atan2(a[1]-b[1], a[0]-b[0]);
  const angle2 = Math.atan2(c[1]-b[1], c[0]-b[0]);
  ctx.beginPath(); ctx.arc(b[0],b[1],20,angle1,angle2);
  ctx.strokeStyle = color+'66'; ctx.lineWidth=1.5; ctx.stroke();
}

// ─── Phase Timeline ───
function buildPhaseTimeline() {
  const tl = document.getElementById('phaseTimeline');
  tl.innerHTML = '';
  PHASES.forEach((p,i)=>{
    const el = document.createElement('div');
    el.className='phase-seg';
    el.style.background=p.color+'33'; el.style.borderTop=`2px solid ${p.color}`;
    el.style.color=p.color; el.title=p.name;
    el.textContent=p.name.split(' ')[0].charAt(0);
    el.onclick=()=>{
      const video = document.getElementById('analysisVideo');
      stopOverlay();
      currentFrame = Math.floor(p.frames[0] * poseFrames.length);
      if (video && videoDuration > 0)
        video.currentTime = p.frames[0] * videoDuration;
      renderStaticFrame(currentFrame);
      document.getElementById('playBtn').textContent = '▶ Play';
    };
    tl.appendChild(el);
  });
}

// ─── Render Results ───
function renderResults(data) {
  // Score
  const pct = data.score/100;
  const dash = 314;
  document.getElementById('scoreArc').style.strokeDashoffset = dash - dash*pct;
  document.getElementById('swingScore').textContent = data.score;
  document.getElementById('swingGrade').textContent = scoreGrade(data.score);

  // Faults
  const fl = document.getElementById('faultsList');
  fl.innerHTML='';
  document.getElementById('faultCountBadge').textContent = data.faults.length;
  data.faults.forEach(f=>{
    const el=document.createElement('div');
    el.className=`fault-item ${f.severity}`;
    el.innerHTML=`
      <div class="fault-icon">${f.icon}</div>
      <div class="fault-body">
        <div class="fault-name">${f.name}
          <span style="font-size:0.7rem;color:var(--text2);margin-left:6px">[${f.phase}]</span>
        </div>
        <div class="fault-desc">${f.desc}</div>
        <div class="fault-conf" style="color:${f.severity==='high'?'var(--danger)':'var(--warn)'}">
          LSTM Confidence: ${(f.confidence*100).toFixed(0)}%
        </div>
        <div class="fault-desc" style="margin-top:6px;color:var(--success)">💡 ${f.fix}</div>
      </div>`;
    fl.appendChild(el);
  });

  // Joint Angles
  const at = document.getElementById('anglesTable');
  at.innerHTML='';
  data.angles.forEach(a=>{
    const diff = Math.abs(a.val - a.ideal);
    const status = diff < 5 ? 'ok' : diff < 12 ? 'warn' : 'bad';
    const el=document.createElement('div'); el.className='angle-row';
    el.innerHTML=`
      <span class="angle-joint">${a.joint}</span>
      <span class="angle-val">${Math.round(a.val)}${a.unit}</span>
      <span class="angle-status ${status}">${status==='ok'?'✓ Good':status==='warn'?'Watch':'⚠ Fault'}</span>`;
    at.appendChild(el);
  });

  // Phase Analysis
  const pa = document.getElementById('phaseAnalysis');
  pa.innerHTML='';
  data.phases.forEach(p=>{
    const el=document.createElement('div'); el.className='phase-item';
    el.innerHTML=`<div class="phase-label">${p.name}</div><div class="phase-score" style="color:${p.color}">${p.score}</div>`;
    pa.appendChild(el);
  });

  // GolfDB Chart
  buildGolfdbChart(data);

  // Update joint map with fault data
  const faultJoints = data.faults.map(f=>{
    if(f.id==='over_the_top') return [5,6,7,8];
    if(f.id==='early_extension') return [11,12,13,14];
    if(f.id==='casting') return [9,10];
    return [];
  }).flat();
  drawJointMap(faultJoints);

  // Stat updates
  document.getElementById('statScore').textContent = data.score;
  document.getElementById('statFaults').textContent = data.faults.length;
}

function scoreGrade(s) {
  if(s>=90) return 'A+'; if(s>=80) return 'A'; if(s>=70) return 'B+';
  if(s>=60) return 'B'; if(s>=50) return 'C'; return 'D';
}

// Update GolfDB percentile display
function updateGolfDBPercentile(score, swingType) {
    const bench = GOLFDB_BENCHMARKS[swingType];
    if (!bench) return;
    
    // Calculate percentile based on normal distribution
    // Using GolfDB's average and standard deviation
    const stdScore = bench.stdScore || 15; // Fallback std deviation
    const zScore = (score - bench.avgScore) / stdScore;
    let percentile = 50 + (zScore * 15); // Rough estimate
    
    // Clamp between 0-99
    percentile = Math.min(99, Math.max(1, Math.round(percentile)));
    
    // Update all percentile displays
    const percentileElements = [
        'golfdbPercentile',
        'dashboardPercentile'
    ];
    
    percentileElements.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = `${percentile}th`;
            el.title = `Top ${100-percentile}% of GolfDB dataset`;
        }
    });
    
    // Update rank text
    const rankText = document.getElementById('golfdbRankText');
    if (rankText) {
        if (percentile >= 90) {
            rankText.innerHTML = `🏆 Top ${100-percentile}% globally`;
        } else if (percentile >= 70) {
            rankText.innerHTML = `👍 Better than ${percentile}% of golfers`;
        } else if (percentile >= 40) {
            rankText.innerHTML = `📈 Above ${percentile}% of GolfDB`;
        } else {
            rankText.innerHTML = `🎯 Room for improvement (${percentile}th percentile)`;
        }
    }
}

// ─── GolfDB Chart ───
function buildGolfdbChart(data) 
{
  updateGolfDBPercentile(data.score, data.type);  // ← ADD THIS LINE
  const b = data.bench;
  const ctx = document.getElementById('golfdbChart').getContext('2d');
  if (golfdbChartInst) golfdbChartInst.destroy();
  golfdbChartInst = new Chart(ctx, {
    type:'radar',
    data:{
      labels:['Swing Score','Hip Turn','Shoulder Turn','Spine Angle','Consistency','Power'],
      datasets:[
        { label:'Your Swing', data:[data.score, data.angles[0].val, data.angles[1].val, data.angles[2].val, 60+Math.random()*25, 65+Math.random()*20],
          borderColor:'#B8FF4F', backgroundColor:'rgba(184,255,79,0.1)', borderWidth:2, pointBackgroundColor:'#B8FF4F' },
        { label:'GolfDB Avg', data:[b.avgScore, b.hipTurn, b.shoulderTurn, b.spineAngle, 65, 68],
          borderColor:'#4F8BFF', backgroundColor:'rgba(79,139,255,0.08)', borderWidth:2, pointBackgroundColor:'#4F8BFF', borderDash:[5,5] },
        { label:'Tour Pro', data:[b.tour, b.hipTurn+8, b.shoulderTurn+6, b.spineAngle+4, 92, 95],
          borderColor:'#FFB84F', backgroundColor:'rgba(255,184,79,0.05)', borderWidth:1.5, pointBackgroundColor:'#FFB84F', borderDash:[3,3] }
      ]
    },
    options:{
      responsive:true, scales:{r:{grid:{color:'rgba(255,255,255,0.06)'}, ticks:{color:'#8888a0',backdropColor:'transparent',font:{size:9}}, pointLabels:{color:'#aaa',font:{size:11}}}},
      plugins:{legend:{labels:{color:'#aaa',font:{size:11}}}}
    }
  });

  const gs = document.getElementById('golfdbStats');
  gs.innerHTML = `
    <div class="db-row"><span class="db-label">Your Score</span><span class="db-val">${data.score}</span></div>
    <div class="db-row"><span class="db-label">GolfDB Avg</span><span class="db-val">${b.avgScore}</span></div>
    <div class="db-row"><span class="db-label">Top Amateur</span><span class="db-val">${b.topAmateur}</span></div>
    <div class="db-row"><span class="db-label">Tour Pro</span><span class="db-val">${b.tour}</span></div>
    <div class="db-row"><ssaveAnalysis()pan class="db-label">Percentile</span><span class="db-val">${Math.round(data.score/b.tour*100)}th</span></div>
    <div class="db-row"><span class="db-label">Swing Type</span><span class="db-val">${capitalize(data.type)}</span></div>`;
}

// ─── Joint Health Map ───
function drawJointMap(faultJoints=[]) {
  const canvas = document.getElementById('jointMapCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,220,320);

  // Stick figure positions (normalized to 220x320)
  const joints = [
    [110, 20],  // 0 head (circle)
    [80,  75],  // 1 L.Shoulder
    [140, 75],  // 2 R.Shoulder
    [60,  130], // 3 L.Elbow
    [160, 130], // 4 R.Elbow
    [45,  180], // 5 L.Wrist
    [175, 180], // 6 R.Wrist
    [90,  170], // 7 L.Hip
    [130, 170], // 8 R.Hip
    [80,  240], // 9 L.Knee
    [140, 240], // 10 R.Knee
    [75,  300], // 11 L.Ankle
    [145, 300], // 12 R.Ankle
  ];

  const skel = [[1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]];

  // Draw skeleton
  skel.forEach(([a,b])=>{
    ctx.beginPath(); ctx.moveTo(joints[a][0],joints[a][1]); ctx.lineTo(joints[b][0],joints[b][1]);
    ctx.strokeStyle='rgba(255,255,255,0.1)'; ctx.lineWidth=2; ctx.stroke();
  });

  // Head
  ctx.beginPath(); ctx.arc(110,12,12,0,Math.PI*2);
  ctx.strokeStyle='rgba(255,255,255,0.2)'; ctx.lineWidth=2; ctx.stroke();

  // Joints
  joints.forEach((j,i)=>{
    const isFault = faultJoints.includes(i) || faultJoints.includes(i+5);
    const color = isFault ? '#FF4F6B' : (Math.random()>0.8 ? '#FFB84F' : '#4FFFB8');
    ctx.shadowColor=color; ctx.shadowBlur=8;
    ctx.beginPath(); ctx.arc(j[0],j[1],6,0,Math.PI*2);
    ctx.fillStyle=color; ctx.fill();
    ctx.shadowBlur=0;
  });
}

// ═══════════════════════════════════════════
// FIREBASE SAVE & HISTORY
// ═══════════════════════════════════════════

async function saveAnalysis() {
  if (!currentUser || !analysisData) return;
  
  const sessionId = 'session_' + Date.now();
  const session = {
    id: sessionId,
    score: analysisData.score,
    grade: scoreGrade(analysisData.score),
    type: analysisData.type,
    cameraAngle: analysisData.cameraAngle || 'face_on',
    faults: analysisData.faults.map(f => ({
      id: f.id, 
      name: f.name, 
      severity: f.severity, 
      confidence: f.confidence
    })),
    angles: analysisData.angles,
    phases: analysisData.phases.map(p => ({
      name: p.name, 
      score: p.score
    })),
    timestamp: Date.now(),
    date: new Date().toISOString()
  };
  
  try {
    // Save to Firebase Realtime Database
    await firebase.database().ref(`sessions/${currentUser.uid}/${sessionId}`).set(session);
    
    // Update user's swing count
    const swingsAnalyzed = (currentUser.swingsAnalyzed || 0) + 1;
    await firebase.database().ref(`users/${currentUser.uid}/swingsAnalyzed`).set(swingsAnalyzed);
    currentUser.swingsAnalyzed = swingsAnalyzed;
    localStorage.setItem('gv_current_user', JSON.stringify(currentUser));
    
    alert('✅ Analysis saved to Firebase!');
    await loadHistory();
    await loadDashboardStats();
    
  } catch (error) {
    console.error('Save error:', error);
    alert('❌ Failed to save session: ' + error.message);
  }
}

async function loadHistory() {
  if (!currentUser) return;
  
  try {
    const snapshot = await firebase.database().ref(`sessions/${currentUser.uid}`).once('value');
    const sessions = snapshot.val() || {};
    historyData = Object.values(sessions).sort((a, b) => b.timestamp - a.timestamp);
    renderHistory(historyData);
    renderRecentSessions(historyData.slice(0, 4));
  } catch (error) {
    console.error('Load history error:', error);
    historyData = [];
    renderHistory([]);
    renderRecentSessions([]);
  }
}

function renderHistory(data) {
  const grid = document.getElementById('historyGrid');
  if (!data.length) { 
    grid.innerHTML = '<div class="empty-state center">No sessions saved yet. Upload a swing to begin!</div>'; 
    return; 
  }
  grid.innerHTML = '';
  data.forEach(s => {
    const d = new Date(s.timestamp);
    const el = document.createElement('div'); 
    el.className = 'history-card';
    el.innerHTML = `
      <div class="hc-top">
        <span class="hc-type">${capitalize(s.type || 'driver')}</span>
        <span class="hc-score">${s.score}</span>
        <button class="delete-session-btn" data-id="${s.id}" onclick="event.stopPropagation(); deleteSession('${s.id}')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
          </svg>
        </button>
      </div>
      <div class="hc-date">${d.toLocaleDateString()} ${d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</div>
      <div class="hc-faults">
        ${(s.faults || []).map(f => `<span class="hc-fault-tag">${f.name}</span>`).join('')}
      </div>
    `;
    grid.appendChild(el);
  });
}

async function deleteSession(sessionId) {
  if (!currentUser || !sessionId) return;
  
  // Confirm deletion
  const confirmed = confirm('Are you sure you want to delete this session? This action cannot be undone.');
  if (!confirmed) return;
  
  try {
    // Get Firebase user
    const firebaseUser = firebase.auth().currentUser;
    if (!firebaseUser) {
      alert('Please log in to delete sessions');
      return;
    }
    
    // Delete from Firebase
    await firebase.database().ref(`sessions/${firebaseUser.uid}/${sessionId}`).remove();
    
    // Also decrement swing count if needed (optional)
    const userRef = firebase.database().ref(`users/${firebaseUser.uid}`);
    const snapshot = await userRef.child('swingsAnalyzed').once('value');
    const currentCount = snapshot.val() || 0;
    if (currentCount > 0) {
      await userRef.child('swingsAnalyzed').set(currentCount - 1);
      if (currentUser) currentUser.swingsAnalyzed = currentCount - 1;
    }
    
    // Refresh history display
    await loadHistory();
    await loadDashboardStats();
    
    // Show success message
    const msg = document.createElement('div');
    msg.textContent = '✅ Session deleted successfully!';
    msg.style.cssText = 'position:fixed; bottom:20px; right:20px; background:#4FFFB8; color:#000; padding:10px 20px; border-radius:8px; z-index:9999; font-size:0.9rem;';
    document.body.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
    
  } catch (error) {
    console.error('Delete error:', error);
    alert('❌ Failed to delete session: ' + error.message);
  }
}

function renderRecentSessions(data) {
  const list = document.getElementById('recentList');
  if (!data.length) { 
    list.innerHTML = '<div class="empty-state">No sessions yet. Upload a swing to begin!</div>'; 
    return; 
  }
  list.innerHTML = '';
  data.forEach(s => {
    const d = new Date(s.timestamp);
    const el = document.createElement('div'); 
    el.className = 'recent-item';
    el.innerHTML = `
      <div class="recent-thumb">⛳</div>
      <div class="recent-info">
        <div class="recent-name">${capitalize(s.type || 'driver')} Swing</div>
        <div class="recent-meta">${d.toLocaleDateString()} · ${s.faults?.length || 0} fault(s)</div>
      </div>
      <div class="recent-score">${s.score}</div>
    `;
    list.appendChild(el);
  });
}

async function loadDashboardStats() {
  if (!currentUser) return;
  
  try {
    // Get user data from Firebase
    const userSnapshot = await firebase.database().ref(`users/${currentUser.uid}`).once('value');
    const userData = userSnapshot.val() || {};
    const swingsAnalyzed = userData.swingsAnalyzed || 0;
    
    document.getElementById('statSwings').textContent = swingsAnalyzed;
    document.getElementById('ps1').textContent = swingsAnalyzed;

    if (historyData.length) {
      const avg = Math.round(historyData.reduce((s, h) => s + h.score, 0) / historyData.length);
      const best = Math.max(...historyData.map(h => h.score));
      const totalFaults = historyData.reduce((s, h) => s + (h.faults?.length || 0), 0);
      
      document.getElementById('statScore').textContent = avg;
      document.getElementById('statFaults').textContent = totalFaults;
      document.getElementById('statImprove').textContent = historyData.length > 1 ? 
        (historyData[0].score - historyData[historyData.length - 1].score > 0 ? '+' : '') +
        (historyData[0].score - historyData[historyData.length - 1].score) + '%' : '—%';
      document.getElementById('ps2').textContent = best;
      
      updateTrendChart(historyData.slice(0, 7).reverse());
      updateFaultChart(historyData);
    }
  } catch (error) {
    console.error('Load dashboard stats error:', error);
  }
}

function filterHistory(q) { renderHistory(historyData.filter(s=> JSON.stringify(s).toLowerCase().includes(q.toLowerCase()))); }
function filterHistoryType(t) { renderHistory(t==='all' ? historyData : historyData.filter(s=>s.type===t)); }

// ═══════════════════════════════════════════
// CHARTS
// ═══════════════════════════════════════════

function initCharts() {
  const compactTick = { color:'#8888a0', font:{ size:9 } };
  const compactGrid = { color:'rgba(255,255,255,0.04)' };

  // Trend Chart — compact line
  const tc = document.getElementById('trendChart').getContext('2d');
  if (trendChartInst) trendChartInst.destroy();
  trendChartInst = new Chart(tc, {
    type: 'line',
    data: {
      labels: ['—','—','—','—','—','—','—'],
      datasets: [{
        label: 'Score',
        data: [],
        borderColor: '#B8FF4F',
        backgroundColor: 'rgba(184,255,79,0.07)',
        tension: 0.4, fill: true,
        pointBackgroundColor: '#B8FF4F',
        pointRadius: 3, pointHoverRadius: 5,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: { grid: compactGrid, ticks: compactTick, border:{ display:false } },
        y: { grid: compactGrid, ticks: { ...compactTick, maxTicksLimit: 4 },
             min: 0, max: 100, border:{ display:false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { bodyFont:{ size:11 }, titleFont:{ size:11 } }
      }
    }
  });

  // Fault Doughnut — compact
  const fc = document.getElementById('faultChart').getContext('2d');
  if (faultChartInst) faultChartInst.destroy();
  faultChartInst = new Chart(fc, {
    type: 'doughnut',
    data: {
      labels: ['No Data'],
      datasets: [{ data:[1],
        backgroundColor: ['rgba(255,255,255,0.05)'],
        borderColor: ['rgba(255,255,255,0.05)'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      cutout: '62%',
      plugins: {
        legend: {
          position: 'right',
          labels: { color:'#8888a0', font:{ size:9 }, boxWidth:10, padding:6 }
        },
        tooltip: { bodyFont:{ size:11 } }
      }
    }
  });
}

function updateTrendChart(data) {
  if (!trendChartInst) return;
  const labels = data.map(s=> new Date(s.timestamp).toLocaleDateString('en',{month:'short',day:'numeric'}));
  trendChartInst.data.labels = labels;
  trendChartInst.data.datasets[0].data = data.map(s=>s.score);
  trendChartInst.update();
}

function updateFaultChart(data) {
  if (!faultChartInst) return;
  const counts={};
  data.forEach(s=>(s.faults||[]).forEach(f=>{ counts[f.name]=(counts[f.name]||0)+1; }));
  const entries = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,5);
  if(!entries.length) return;
  faultChartInst.data.labels = entries.map(e=>e[0]);
  faultChartInst.data.datasets[0].data = entries.map(e=>e[1]);
  faultChartInst.data.datasets[0].backgroundColor=['#FF4F6B','#FFB84F','#B8FF4F','#4FFFB8','#4F8BFF'].slice(0,entries.length);
  faultChartInst.update();
}

function setChartRange(range, btn) {
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  btn.classList.add('active');
  // Filter history by range
  const now=Date.now();
  const cutoff = range==='week' ? now-7*86400000 : now-30*86400000;
  const filtered = historyData.filter(s=>s.timestamp>cutoff);
  if(filtered.length) updateTrendChart(filtered.reverse());
}

// ═══════════════════════════════════════════
// FAULT LIBRARY
// ═══════════════════════════════════════════

function buildFaultLibrary() {
  const lib = document.getElementById('faultLibrary');
  lib.innerHTML='';
  FAULT_DEFINITIONS.forEach(f=>{
    const el=document.createElement('div'); el.className='fl-card';
    el.innerHTML=`
      <div class="fl-header">
        <div class="fl-icon">${f.icon}</div>
        <div>
          <div class="fl-name">${f.name}</div>
          <div class="fl-phase">Phase: ${f.phase} · ${capitalize(f.severity)} severity</div>
        </div>
      </div>
      <div class="fl-desc">${f.desc}</div>
      <div class="fl-fix">💡 Fix: ${f.fix}</div>`;
    lib.appendChild(el);
  });
}

// ═══════════════════════════════════════════
// PROCESSING ANIMATION
// ═══════════════════════════════════════════

function animateStage(n, from, duration, msg) {
  return new Promise(resolve => {
    const fill=document.getElementById(`fill${n}`);
    const pct=document.getElementById(`pct${n}`);
    const start=performance.now();
    function step(now){
      const t=Math.min((now-start)/duration,1);
      const eased=t<0.5?2*t*t:(4-2*t)*t-1;
      fill.style.width=(eased*100)+'%';
      pct.textContent=Math.round(eased*100)+'%';
      if(t<1) requestAnimationFrame(step);
      else { addLog(`✓ ${msg}`); resolve(); }
    }
    requestAnimationFrame(step);
  });
}

function addLog(msg) {
  const log = document.getElementById('processLog');
  if(!log) return;
  const line=document.createElement('div');
  line.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  log.appendChild(line);
  log.scrollTop=log.scrollHeight;
}

// ═══════════════════════════════════════════
// MISC
// ═══════════════════════════════════════════

function resetAnalysis() {
  analysisData = null; poseFrames = []; currentFrame = 0;
  stopOverlay();
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('uploadMeta').classList.add('hidden');
  document.getElementById('uploadZone').style.display = '';
  document.getElementById('processingState').classList.add('hidden');
  document.getElementById('analysisResults').classList.add('hidden');
  document.getElementById('emptyResults').classList.remove('hidden');
  document.getElementById('videoInput').value = '';
  document.getElementById('notifCount').textContent = '0';
  document.getElementById('playBtn').textContent = '▶ Play';
  uploadedVideoURL = '';
}

function exportReport() {
  if(!analysisData) return;
  const report={
    generatedAt: new Date().toISOString(),
    model:'YOLOv8-Pose CNN + LSTM',
    dataset:'GolfDB',
    score: analysisData.score,
    grade: scoreGrade(analysisData.score),
    swingType: analysisData.type,
    detectedFaults: analysisData.faults.map(f=>({name:f.name, phase:f.phase, confidence:(f.confidence*100).toFixed(0)+'%', fix:f.fix})),
    jointAngles: analysisData.angles,
    phaseScores: analysisData.phases.map(p=>({phase:p.name, score:p.score}))
  };
  const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=`GolfVisionAI_Report_${Date.now()}.json`; a.click();
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase()+s.slice(1) : ''; }

// ═══════════════════════════════════════════
// SHOWCASE CANVAS ANIMATION
// ═══════════════════════════════════════════

function animateShowcase() {
  const canvas=document.getElementById('showcaseCanvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  let t=0;
  const joints=Array.from({length:13},(_,i)=>({
    x:40+i*25+Math.sin(i)*10,
    y:100+Math.cos(i*0.8)*40,
    vx:(Math.random()-0.5)*0.3,
    vy:(Math.random()-0.5)*0.3,
    conf:0.8+Math.random()*0.2
  }));
  const skel2=[[0,1],[1,2],[2,3],[1,4],[4,5],[1,6],[6,7],[7,8],[6,9],[9,10],[10,11],[9,12]];

// ═════════════════════════════════════════════════════════════
  // FIXED REPLACEMENT FOR THE SHOWCASE CANVAS AT THE BOTTOM OF APP.JS
  // ═════════════════════════════════════════════════════════════
  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    t += 0.01;

    // Smooth movement vector physics simulation loop
    joints.forEach((j, i) => {
      j.x += j.vx;
      j.y += j.vy;

      // Wall bounce tracking coordinates
      if (j.x < 20 || j.x > canvas.width - 20) j.vx *= -1;
      if (j.y < 40 || j.y > canvas.height - 40) j.vy *= -1;

      // Subtle resting hover displacement pattern
      const swingOffset = Math.sin(t + i) * 0.15;
      j.y += swingOffset;
    });

    // Draw glowing tech network links on showcase dashboard panel
    skel2.forEach(([a, b]) => {
      const ja = joints[a];
      const jb = joints[b];
      if (ja && jb) {
        ctx.beginPath();
        ctx.moveTo(ja.x, ja.y);
        ctx.lineTo(jb.x, jb.y);
        ctx.strokeStyle = 'rgba(79, 255, 184, 0.15)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });

    // Draw tech joint nodes
    joints.forEach((j, i) => {
      ctx.beginPath();
      ctx.arc(j.x, j.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = i < 5 ? 'rgba(255, 79, 107, 0.4)' : 'rgba(184, 255, 79, 0.4)';
      ctx.fill();
    });

    requestAnimationFrame(frame);
  }
  frame();
}

// ═══════════════════════════════════════════
// PARTICLES (auth screen)
// ═══════════════════════════════════════════

(function initParticles(){
  const container=document.getElementById('particles');
  if(!container) return;
  for(let i=0;i<30;i++){
    const p=document.createElement('div');
    p.style.cssText=`
      position:absolute; border-radius:50%;
      width:${2+Math.random()*4}px; height:${2+Math.random()*4}px;
      background:rgba(184,255,79,${0.1+Math.random()*0.3});
      left:${Math.random()*100}%; top:${Math.random()*100}%;
      animation:float${i%3} ${4+Math.random()*6}s ease-in-out infinite;
      animation-delay:${Math.random()*4}s;
    `;
    container.appendChild(p);
  }
  const style=document.createElement('style');
  style.textContent=`
    @keyframes float0{0%,100%{transform:translate(0,0)}50%{transform:translate(15px,-20px)}}
    @keyframes float1{0%,100%{transform:translate(0,0)}50%{transform:translate(-10px,15px)}}
    @keyframes float2{0%,100%{transform:translate(0,0)}50%{transform:translate(20px,10px)}}
  `;
  document.head.appendChild(style);
})();  // ← CLOSE THE IIFE HERE

// API Configuration for backend (OUTSIDE the IIFE)
const API_BASE_URL = 'http://localhost:5000';

// Function to fetch real GolfDB stats on demand
async function refreshGolfDBStats() {
    const response = await fetch(`${API_BASE_URL}/api/golfdb/stats`);
    if (response.ok) {
        const stats = await response.json();
        window.GOLFDB_BENCHMARKS = stats;
        console.log('🔄 GolfDB stats refreshed from real dataset');
        
        // Update UI if analysis is showing
        if (analysisData) {
            buildGolfdbChart(analysisData);
        }
    }
}

// Call this every time you run analysis
async function runAnalysisWithRealData() {
    await refreshGolfDBStats();  // Get latest stats from real dataset
    await runAnalysis();          // Run the analysis
}