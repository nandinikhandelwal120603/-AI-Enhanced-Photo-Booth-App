let total = window.totalImages || 1;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function startCapture() {
    for (let i = 0; i < total; i++) {
        await runCountdown();
        await takePicture();
    }
    window.location.href = "/finalize";
}

async function runCountdown() {
    const countdown = document.getElementById('countdown');
    countdown.style.display = 'block';
    for (let i = 5; i > 0; i--) {
        countdown.innerText = i;
        await sleep(1000);
    }
    countdown.style.display = 'none';
}

async function takePicture() {
    try {
        const response = await fetch('/take_picture', {
            method: 'POST'
        });

        if (!response.ok) {
            console.error("Failed to capture image.");
        } else {
            console.log("Captured image successfully!");
        }
    } catch (err) {
        console.error("Error during capture:", err);
    }
}
