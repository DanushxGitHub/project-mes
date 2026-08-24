document.addEventListener("DOMContentLoaded", () => {

  const SCORE_ON_AGREE = { 1: true, 2: false, 3: false, 4: false, 5: true, 6: true, 7: false, 8: false, 9: true, 10: false };
  const AGREE_VALUES = new Set(["da", "sa"]);

  const scoreChip = document.getElementById("liveScore");
  const scoreValue = document.getElementById("scoreValue");
  const form = document.getElementById("assessmentForm");

  function readScore() {
    let total = 0;
    for (let i = 1; i <= 10; i++) {
      const checked = document.querySelector(`input[name="q${i}"]:checked`);
      if (!checked) continue;
      const agrees = AGREE_VALUES.has(checked.value);
      if (agrees === SCORE_ON_AGREE[i]) total += 1;
    }
    return total;
  }

  function updateScore() {
    if (!scoreChip || !scoreValue) return;
    scoreChip.hidden = false;
    scoreValue.textContent = String(readScore());
  }

  if (form) {
    form.querySelectorAll('input[type="radio"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        updateScore();
        const alertBox = document.querySelector(".alert-error");
        if (alertBox) alertBox.remove();
      });
    });

    form.addEventListener("submit", (event) => {
      if (event.submitter && event.submitter.hasAttribute("formaction")) {
        return;
      }

      let missing = 0;
      for (let i = 1; i <= 10; i++) {
        if (!document.querySelector(`input[name="q${i}"]:checked`)) missing += 1;
      }

      if (missing > 0) {
        event.preventDefault();
        let alertBox = document.querySelector(".alert-error");
        if (!alertBox) {
          alertBox = document.createElement("div");
          alertBox.className = "alert-error";
          alertBox.setAttribute("role", "alert");
          form.prepend(alertBox);
        }
        alertBox.textContent =
          "Please answer all ten screening questions before submitting.";
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    });
  }

  updateScore();

  const confidenceFill = document.getElementById("confidenceFill");
  if (confidenceFill) {
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${confidenceFill.dataset.width}%`;
    });
  }

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  document.querySelectorAll(".reveal").forEach((element) => {
    revealObserver.observe(element);
  });
});
