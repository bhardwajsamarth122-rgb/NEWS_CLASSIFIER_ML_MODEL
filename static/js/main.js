function toggleDarkMode() {
    document.body.classList.toggle("dark");
}

function showSpinner() {
    document.getElementById("spinner").style.display = "block";
}

function insertSample(text) {
    document.getElementById("newsText").value = text;
}
