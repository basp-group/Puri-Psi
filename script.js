
// Function to redirect after 30 seconds
function redirectToNewPage() {
    window.location.href = "https://basp-group.github.io/BASPLib/index.html";
}

// Function to start the countdown and redirect
function startCountdown() {
    var countdown = 5; // Countdown time in seconds
    var countdownInterval = setInterval(function() {
        document.getElementById('countdown').innerText = countdown + ' sec.';
        countdown--;
        if (countdown < 0) {
            clearInterval(countdownInterval);
            redirectToNewPage();
        }
    }, 1000); // Update countdown every second
}

// Call startCountdown when the page loads
window.onload = startCountdown;
