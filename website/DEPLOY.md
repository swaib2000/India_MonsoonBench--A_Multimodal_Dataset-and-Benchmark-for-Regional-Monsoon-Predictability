# Deploying the India MonsoonBench Website

This `website/` directory is a self-contained static artifact preview. It can
be deployed directly to Netlify, GitHub Pages, or any static hosting service.

## Recommended Netlify Deployment

1. Log in to Netlify.
2. Choose **Add new site**.
3. Choose **Deploy manually** if you want the fastest route.
4. Drag and drop the entire `website/` folder.
5. Netlify will publish the site and give you a temporary URL.
6. Rename the site to something descriptive, for example:

```text
india-monsoonbench
```

The deploy directory is:

```text
/home/home2/PhD/2025_06_17_Data_Weather_Analytics_Paper/website
```

No build command is required.

## Optional Git-Based Netlify Deployment

If you push this folder to a GitHub repository, configure Netlify as:

```text
Build command:  leave empty
Publish directory: website
```

## What Is Included

- `index.html`: landing page and project framing.
- `dataset.html`: modalities, target labels, regional grouping, limitations.
- `benchmarks.html`: benchmark table and diagnostic explanation.
- `gallery.html`: high-resolution regional, state, and correlation figures.
- `reproducibility.html`: commands used for extraction, splitting, and training.
- `assets/images/`: copied high-resolution figures.
- `assets/tables/`: CSV summaries from completed runs.

## Important Rebuttal Note

The website is useful as an artifact preview, but the rebuttal itself should be
self-contained because conference rebuttal systems may not allow or consider
external hyperlinks. Put the key tables and numerical claims directly in the
response, and use the website as supporting material or for camera-ready release.

## Final Artifact Recommendation

For the camera-ready or public release, pair this page with an archival dataset
location such as Zenodo, OSF, institutional storage, or a DOI-backed release.
Large patch datasets should not be hosted directly inside the static website.
