function trainButtonClicked() {
    console.log("Training started");
    const trainButton = document.getElementById("trainButton");
    let enabled = trainButton.style.display;

    if (enabled === "none") {
        trainButton.style.display = "block";
    }
}