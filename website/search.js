(function () {
  function setupSearch(inputId, countId) {
    const input = document.getElementById(inputId);
    const counter = document.getElementById(countId);
    const cards = Array.from(document.querySelectorAll(".searchable"));

    if (!input || !counter || cards.length === 0) {
      return;
    }

    function update() {
      const query = input.value.trim().toLowerCase();
      let shown = 0;

      cards.forEach((card) => {
        const haystack = (card.dataset.searchText || card.textContent || "").toLowerCase();
        const visible = !query || haystack.includes(query);
        card.classList.toggle("is-hidden", !visible);
        if (visible) {
          shown += 1;
        }
      });

      counter.textContent = query
        ? `Showing ${shown} matching section${shown === 1 ? "" : "s"} for "${query}".`
        : "Showing all searchable items.";
    }

    input.addEventListener("input", update);
    update();
  }

  setupSearch("dataset-search", "dataset-search-count");
  setupSearch("tutorial-search", "tutorial-search-count");
})();
